"""Aivara application factory."""
import os
from flask import Flask, send_from_directory

from .config import Config
from .db import close_db, init_db
from .routes import api


def create_app(overrides=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder=None)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)
    app.teardown_appcontext(close_db)
    app.register_blueprint(api, url_prefix="/api")
    static = os.path.join(os.path.dirname(__file__), "..", "static")

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.after_request
    def secure_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'")
        return resp

    with app.app_context():
        init_db()
    return app
