import logging
from flask import Flask, jsonify
from routes.image_routes import images
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.register_blueprint(images)

@app.errorhandler(413)
def too_large(_error):
    return jsonify(ok=False, error={"code": "upload_too_large", "message": "Tệp vượt quá giới hạn tải lên 20 MB."}), 413

@app.errorhandler(500)
def internal_error(_error):
    app.logger.exception("Unhandled server error")
    return jsonify(ok=False, error={"code": "internal_error", "message": "Lỗi máy chủ khi xử lý yêu cầu."}), 500

if not app.debug:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
