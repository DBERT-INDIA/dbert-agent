import os
import time
import logging
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import winsound
except ImportError:
    winsound = None

from src.voice.stt_whisper import transcribe
from src.voice.tts_piper import synthesize

logger = logging.getLogger("dbert.voice.voice_controller")

def record_voice(output_path: str, threshold: float = 500.0, silence_limit: float = 1.5, samplerate: int = 16000) -> bool:
    """
    Records audio from the microphone until silence is detected using an RMS energy threshold.
    """
    if sd is None:
        logger.error("sounddevice is not available. Cannot record voice.")
        print("[Microphone hardware/library missing. Voice recording disabled.]")
        return False
        
    audio_data = []
    silent_chunks = 0
    chunk_duration = 0.1  # 100ms
    chunk_samples = int(samplerate * chunk_duration)
    
    max_silent_chunks = int(silence_limit / chunk_duration)
    speaking_started = False
    
    max_duration = 30.0
    start_time = time.time()
    
    logger.info("Starting audio stream recording...")
    print("\n[Listening... Speak now. Pause 1.5s to finish.]")
    
    def callback(indata, frames, time_info, status):
        if status:
            logger.warning(status)
        audio_data.append(indata.copy())
        
    no_speech_timeout = 5.0
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=chunk_samples, dtype='int16'):
            while time.time() - start_time < max_duration:
                sd.sleep(100)
                
                if not speaking_started and (time.time() - start_time > no_speech_timeout):
                    break
                    
                if len(audio_data) == 0:
                    continue
                    
                latest_chunk = audio_data[-1]
                if len(latest_chunk) > 0:
                    rms = np.sqrt(np.mean(latest_chunk.astype(np.float32)**2))
                else:
                    rms = 0.0
                    
                if rms > threshold:
                    if not speaking_started:
                        speaking_started = True
                        print("[Voice active...]")
                    silent_chunks = 0
                else:
                    if speaking_started:
                        silent_chunks += 1
                        if silent_chunks >= max_silent_chunks:
                            break
    except Exception as e:
        logger.error(f"Failed to record audio from device: {e}")
        print(f"[Microphone hardware error: {e}]")
        return False
        
    if not speaking_started or len(audio_data) == 0:
        print("[No speech detected.]")
        return False
        
    recording = np.concatenate(audio_data, axis=0)
    sf.write(output_path, recording, samplerate)
    return True

def speak_text(text: str, config_manager, provider_manager, active_model) -> None:
    """
    Synthesizes speech from text and plays it back on Windows using winsound.
    """
    voice_pref = config_manager.config.get("voice", {}).get("piper_voice", "en_US-lessac-medium")
    
    wav_path = None
    try:
        wav_path = synthesize(
            text=text,
            voice=voice_pref,
            config_manager=config_manager,
            provider_manager=provider_manager
        )
        if wav_path and os.path.exists(wav_path):
            if winsound:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            else:
                logger.warning("winsound is not available. Skipping playback.")
    except Exception as e:
        logger.error(f"Speech playback failed: {e}")
        raise e
    finally:
        # Cleanup output wav file
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as ex:
                logger.debug(f"Failed to clean up audio file {wav_path}: {ex}")

def start_voice_loop(active_session, active_model, provider_manager, config_manager, session_manager) -> None:
    """
    Enters a continuous conversation voice loop (record -> STT -> LLM -> TTS -> play).
    """
    print("\n" + "="*50)
    print("             DBERT COLLABORATIVE VOICE MODE            ")
    print("="*50)
    print(" - Speak naturally. Say 'exit' or 'stop' to end.")
    print(" - Press Ctrl+C at any time to interrupt.")
    print("="*50 + "\n")
    
    home_dir = Path.home() / ".dbert"
    rec_path = str(home_dir / "tmp" / "user_voice.wav")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    
    error_count = 0
    while True:
        try:
            # 1. Record voice
            success = record_voice(rec_path, threshold=400.0, silence_limit=1.5)
            if not success:
                error_count += 1
                if error_count >= 3:
                    print("[Too many consecutive audio errors. Exiting Voice Mode.]")
                    break
                time.sleep(0.5)
                continue
                
            error_count = 0  # Reset on success
                
            # 2. Transcribe WAV
            try:
                user_text = transcribe(
                    audio_path=rec_path,
                    model_size="base",
                    config_manager=config_manager,
                    provider_manager=provider_manager
                )
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                print(f"[Error transcribing audio: {e}]")
                continue
            finally:
                if os.path.exists(rec_path):
                    try:
                        os.remove(rec_path)
                    except Exception:
                        pass
                        
            print(f"\nYou (Voice): {user_text}")
            
            if user_text.strip().lower() in ["exit", "stop", "exit voice", "stop voice"]:
                print("[Exiting Voice Mode]")
                break
                
            if not user_text.strip():
                continue
                
            # 3. Call LLM turn
            query_emb = None
            try:
                from src.rag.ingest import get_embeddings
                query_emb = get_embeddings([user_text], active_model, provider_manager)[0]
            except Exception as e:
                logger.warning(f"Could not generate query embedding: {e}")
                
            session_manager.append_message(active_session.id, "user", user_text, embedding=query_emb)
            
            # Ground document RAG
            context_docs = []
            if query_emb:
                try:
                    from src.rag.vector_store import VectorStore
                    vs = VectorStore(active_session.workspace_id, config_manager.app_dir)
                    matches = vs.query(query_emb, top_k=3)
                    for doc_path, chunk_text, meta, score in matches:
                        if score > 0.35:
                            context_docs.append(f"Source: {Path(doc_path).name} (Score: {score:.2f})\n{chunk_text}")
                except Exception as e:
                    logger.error(f"Error querying document context: {e}")
                    
            # Ground past history RAG
            context_history = []
            if query_emb:
                try:
                    from src.memory.history_search import semantic_search_history
                    history_matches = semantic_search_history(
                        session_manager.db_path,
                        query_emb,
                        active_session.workspace_id,
                        top_k=3
                    )
                    for match in history_matches:
                        if match.similarity > 0.4 and match.content.strip().lower() != user_text.strip().lower():
                            context_history.append(f"Past Turn ({match.role}): {match.content}")
                except Exception as e:
                    logger.error(f"Error querying history context: {e}")
                    
            grounded_input = ""
            if context_docs:
                grounded_input += "\n[CONTEXT FROM LOCAL DOCUMENTS]\n" + "\n---\n".join(context_docs) + "\n"
            if context_history:
                grounded_input += "\n[CONTEXT FROM PAST CONVERSATIONS]\n" + "\n---\n".join(context_history) + "\n"
                
            if grounded_input:
                grounded_input += f"\nUser Query: {user_text}\n"
            else:
                grounded_input = user_text
                
            litellm_messages = []
            litellm_messages.append({"role": "system", "content": "You are DBERT, a local-first privacy-first AI desktop assistant. Keep responses brief."})
            
            # Get rolling history from session
            messages_history = session_manager.get_session_messages(active_session.id)
            for msg in messages_history[-10:-1]:
                litellm_messages.append({"role": msg.role, "content": msg.content})
                
            litellm_messages.append({"role": "user", "content": grounded_input})
            
            print("DBERT thinking...")
            from src.main import execute_completion_with_fallback
            assistant_reply, answered_model = execute_completion_with_fallback(
                active_model,
                litellm_messages,
                provider_manager,
                config_manager,
                workspace_id=active_session.workspace_id
            )
            
            print(f"DBERT: {assistant_reply}")
            
            # 4. Speak response
            try:
                speak_text(assistant_reply, config_manager, provider_manager, active_model)
            except Exception as e:
                logger.error(f"Voice playback failed: {e}")
                print(f"[Voice synth error: {e}]")
                
            # Save assistant reply and embed
            reply_emb = None
            try:
                from src.rag.ingest import get_embeddings
                reply_emb = get_embeddings([assistant_reply], active_model, provider_manager)[0]
            except Exception as e:
                logger.warning(f"Could not embed response: {e}")
                
            session_manager.append_message(active_session.id, "assistant", assistant_reply, embedding=reply_emb)
            
        except KeyboardInterrupt:
            print("\n[Voice Mode interrupted. Returning to CLI text mode.]")
            break
        except Exception as e:
            logger.error(f"Voice turn loop exception: {e}")
            print(f"[Voice loop turn failed: {e}]")
            time.sleep(1)
