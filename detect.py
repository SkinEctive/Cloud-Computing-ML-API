from flask import Flask, request, jsonify
import mysql.connector
import os
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_hub as hub
from google.cloud import storage
import numpy as np
from io import BytesIO
from nanoid import generate

app = Flask(__name__)
load_dotenv(dotenv_path='./.env')
model = load_model('model96.h5', compile=False, custom_objects={
                   'KerasLayer': hub.KerasLayer})

# Mendapatkan path ke file keyfile.json yang berada di direktori yang sama
serviceKeyPath = os.path.join(os.path.dirname(__file__), 'service.json')
client = storage.Client.from_service_account_json(serviceKeyPath)

# Define the labels
labels = [
    "Cellulitis",
    "Impetigo",
    "Athlete Foot",
    "Nail Fungus",
    "Ringworm",
    "Cutaneous Larva Migrans",
    "Chickenpox",
    "Shingles"
]

# Fungsi untuk menghubungkan ke database MySQL


def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'),
    )


def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/detect/<userId>', methods=['POST'])
def detect(userId):
    if 'file' not in request.files:
        response = jsonify(
            {
                "status": False,
                "message": "No file part in the request.",
            }
        )
        response.status_code = 400  # HTTP 400 Bad Request
        return response

    file = request.files['file']
    if file.filename == '':
        response = jsonify(
            {
                "status": False,
                "message": "No selected file.",
            }
        )
        response.status_code = 400  # HTTP 400 Bad Request
        return response

    try:
        # Read the file into a BytesIO object
        file_stream = BytesIO()
        file.save(file_stream)
        file_stream.seek(0)

        image = Image.open(file_stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_label = labels[np.argmax(predictions)]

        detectHistoryId = generate(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', size=16)

        fileNameRandomizer = generate(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', size=16)

        blob_name = f"detectHistoryImage/{fileNameRandomizer}"

        bucket = client.bucket(os.getenv('BUCKET_NAME'))
        blob = bucket.blob(blob_name)
        file_stream.seek(0)
        blob.upload_from_file(file_stream, content_type=file.content_type)
        history_img_url = blob.public_url

        conn = get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT diseaseId, diseaseName, diseaseDescription, diseaseAction FROM Disease WHERE diseaseName = %s", (predicted_label,))
        diseaseResult = cursor.fetchone()

        if not diseaseResult:
            response = jsonify(
                {
                    "status": False,
                    "message": "Disease not found in database.",
                }
            )
            response.status_code = 404  # HTTP 404 Not Found
            return response

        disease_id, disease_name, disease_action, disease_description = diseaseResult

        # Insert detection history into DetectHistory table
        cursor.execute(
            "INSERT INTO DetectHistory (detectHistoryId, userId, diseaseId, historyImgUrl, createdAt) VALUES (%s, %s, %s, %s, %s)",
            (detectHistoryId, userId, disease_id,
             history_img_url, datetime.now())
        )
        conn.commit()
        cursor.close()
        conn.close()

        response = jsonify(
            {
                "status": True,
                "message": "Disease Detected",
                "data": {
                    "detectHistoryId": detectHistoryId,
                    "diseaseName": disease_name,
                    "diseaseAction": disease_action,
                    "diseaseDescription": disease_description,
                }
            }
        )
        response.status_code = 200  # HTTP 200 OK
        return response

    except Exception as e:
        response = jsonify(
            {
                "status": False,
                "message": str(e),
            }
        )
        response.status_code = 500  # HTTP 500 Internal Server Error
        return response


# Get all detections
@app.route('/detect/history', methods=['GET'])
def get_detection_history():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Execute SQL query to fetch all detection history records
        cursor.execute("SELECT * FROM DetectHistory")
        history_records = cursor.fetchall()

        # Close cursor and connection
        cursor.close()
        conn.close()

        # Format the results as JSON using jsonify
        history_data = []
        for record in history_records:
            history_data.append({
                'detectHistoryId': record[0],
                'userId': record[1],
                'diseaseId': record[2],
                'historyImgUrl': record[3],
                # Format datetime as string
                'createdAt': record[4].strftime('%Y-%m-%d %H:%M:%S')
            })

        return jsonify({
            'status': True,
            'message': 'Detection history retrieved successfully.',
            'data': history_data
        }), 200

    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500


# Get predictions by id
@app.route("/detect/history/<userId>", methods=["GET"])
def getDetectHistoryById(userId):
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Check if the userId exists in the User database
        cursor.execute(
            "SELECT * FROM User WHERE userId = %s", (userId,))
        user_record = cursor.fetchone()

        if not user_record:
            cursor.close()
            conn.close()
            return jsonify({
                'status': False,
                'message': f'User with ID {userId} not found.',
            }), 404  # HTTP 404 Not Found

        # Now check if the userId has any detection history
        cursor.execute(
            """SELECT dh.detectHistoryId, dh.userId, dh.diseaseId, dh.historyImgUrl, dh.createdAt, d.diseaseName, d.diseaseAction, d.diseaseDescription
               FROM DetectHistory dh
               JOIN Disease d ON dh.diseaseId = d.diseaseId
               WHERE dh.userId = %s""", (userId,))
        history_records = cursor.fetchall()

        cursor.close()
        conn.close()

        if not history_records:
            return jsonify({
                'status': False,
                'message': f'User with ID {userId} has no detection history.',
                'data': []
            }), 404  # HTTP 404 Not Found

        history_data = []
        for record in history_records:
            history_data.append({
                'detectHistoryId': record[0],
                'userId': record[1],
                'diseaseId': record[2],
                'historyImgUrl': record[3],
                'createdAt': record[4].strftime('%Y-%m-%d %H:%M:%S'),
                'diseaseName': record[5],
                'diseaseAction': record[6],
                'diseaseDescription': record[7]
            })

        return jsonify({
            'status': True,
            'message': f'Detection history for userId {userId} retrieved successfully.',
            'data': history_data
        }), 200

    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500


@app.route("/")
def home():
    response = jsonify(
        {
            "message": "Welcome to SkinEctive API Machine Learning Server."
        }
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv("PORT"))
