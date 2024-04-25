from flask import Flask, flash, jsonify, request,send_file, redirect, url_for, redirect, render_template, send_from_directory
from dotenv import load_dotenv
import os, sys
import subprocess
import datetime
import threading
import logging
import uuid
import time
from werkzeug.utils import secure_filename
import json
from flask_cors import CORS
import sqlite3
import time
import zipfile
from process import main

logger = logging.getLogger('webserver')
logging.basicConfig(level=logging.INFO,
    format='%(name)-10s %(levelname)-8s [%(asctime)s] %(message)s',
)

load_dotenv()
PATHBASE = os.path.abspath(os.path.dirname(__file__))
logger.info(f"Base Path : {PATHBASE}")

if 'uploads' not in os.listdir(os.path.join(PATHBASE, "static")):
    print("Creating Upload directory.....")
    os.makedirs('static/uploads')
else:
    for file in os.listdir(os.path.join(PATHBASE, "static/uploads")):
        file_path = os.path.join(os.path.join(PATHBASE, "static/uploads"), file)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

if 'converted' not in os.listdir(os.path.join(PATHBASE, "static")):
    print("Creating Converted directory.....")
    os.makedirs('static/converted')
else:
    for file in os.listdir(os.path.join(PATHBASE, "static/converted")):
        file_path = os.path.join(os.path.join(PATHBASE, "static/converted"), file)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

if os.path.exists(os.path.join(PATHBASE, "instance/database.db")):
    os.remove(os.path.join(PATHBASE, "instance/database.db"))


app = Flask(__name__)

CORS(app)
# sets max payload limit 
app.config['MAX_CONTENT_LENGTH'] = 200 * 1000 * 1000

# setup and start the flask app
app = Flask(__name__)
app.app_context().push()


try :
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()
    dbcurs.execute("SELECT * FROM user")
    dbconn.close()
except sqlite3.OperationalError :
    logger.warning("DB : Creating new database")
    open(os.getenv('DATABASE_PATH'),'w').close()
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()
    dbcurs.execute("""
CREATE TABLE user (
        user_uuid VARCHAR(36) NOT NULL, 
        file_uuid VARCHAR(36) NOT NULL,
        name VARCHAR(100) NOT NULL,
        path VARCHAR(100) NOT NULL, 
        created_at DATETIME DEFAULT (CURRENT_TIMESTAMP), 
        status VARCHAR(100) DEFAULT 'Pending' NOT NULL, 
        processed_path VARCHAR(100) DEFAULT 'NaN' NOT NULL, 
        expiry INT NULL,
        PRIMARY KEY (user_uuid, file_uuid)
);
""")
    dbconn.commit()
    dbconn.close()

DBSCHEMA = {
    "user_uuid" : 0,
    "file_uuid" : 1,
    "name" : 2,
    "path" : 3,
    "created_at" : 4,
    "status" : 5,
    "processed_path" : 6,
    "expiry" : 7
}

# ***************************************
# Helper Functions
# ***************************************



# ***************************************
# BEGIN server route definitions
# ***************************************

@app.route('/')
def landing_page():
    """Home page. User can upload files from here"""
    return render_template('landing.html')
    

@app.route('/upload', methods=['POST','GET'])
def upload_page():

    if request.method == 'POST':
        files = {}
        if len(request.files) == 0:
            flash('No file part')
            return render_template('landing.html')
        

        for f in request.files.keys():
            # extract name of file
            filename = secure_filename(request.files[f].filename)
            uuid_file = f.split("_")[1]
            files[request.files[f]] = uuid_file

        # unique user id
        user_uuid = str(uuid.uuid1())

        for f in files.keys():
            dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
            dbcurs = dbconn.cursor()
            # extract name of file
            filename = f.filename
            ext = filename.split('.')[1]
            # new name
            filename = str(files[f]) + "_" + filename
            # saving files locally
            path =  os.path.join("static/uploads", filename)
            f.save(path)  

            id = str(files[f])  # file id
            current_time = time.time()

            dbcurs.execute("""
            INSERT INTO user(user_uuid,file_uuid,name,path,expiry) VALUES (?,?,?,?,?) 
            """, (user_uuid, id, filename, path, current_time))
            dbconn.commit()
            dbconn.close()

            logger.info(f"Created file {id} with user-id {user_uuid}")

        t = threading.Thread(target=main)
        t.start()

        return redirect(url_for('.loader_page',user_uuid=user_uuid)) 
        # user is directed to /display and using AJAX, converted files are displayed
    else:
        """Home page. User can upload files from here"""
        return render_template('landing.html')

@app.route('/loader')
def loader_page():
    """Display page to download files"""
    user_uuid = request.args['user_uuid']
    return render_template('loader.html', user_uuid=user_uuid)

@app.route('/display')
def display_page():
    """Display page to download files"""
    user_uuid = request.args['user_uuid']
    return render_template('display.html', user_uuid=user_uuid)

@app.route('/getdata/<id>')
def get_user_data(id):
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()
    query = dbcurs.execute(f'SELECT * FROM user WHERE user_uuid="{id}"').fetchall()
    response = []
    dbconn.commit()
    dbconn.close()

    for file in query:
        response.append({
            'user_id': file[DBSCHEMA["user_uuid"]],
            'file_id': file[DBSCHEMA["file_uuid"]],
            'path': file[DBSCHEMA["path"]],
            'status': file[DBSCHEMA['status']],
            'name': file[DBSCHEMA["name"]],
            'processed_path': file[DBSCHEMA["processed_path"]]
        })

    return jsonify(response)


@app.route('/status/<id>')
def status_check(id):
    """Return JSON with info about whether the uploaded file has been parsed successfully."""
    # query = User.query.filter(User.user_uuid == id).all()
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()
    query = dbcurs.execute(f'SELECT * FROM user WHERE user_uuid="{id}"').fetchall()
    response = []
    dbconn.commit()
    dbconn.close()

    for file in query:
        fstatus = file[DBSCHEMA["status"]]
        if fstatus == 'Done':
            message = 'File parsed successfully.'
        elif fstatus == 'Pending':
            message = 'File parsing pending.'
        else:
            message = 'File parsing failed.'

        response.append({
            'user_id': file[DBSCHEMA["user_uuid"]],
            'file_id': file[DBSCHEMA["file_uuid"]],
            'status': fstatus,
            'name': file[DBSCHEMA["name"]],
            'message': message,
            'processed_path': file[DBSCHEMA["processed_path"]]
        })
    return jsonify(response)


@app.route('/download/<id>')
def download_file(id):
    """Download the converted file."""
    print("download link")
    outputFile = os.path.join(PATHBASE, f'static/converted/{id}')
    return send_file(outputFile, as_attachment=True)


@app.route('/downloadAll/<id>')
def download_all(id):
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()
    query = dbcurs.execute(f'SELECT * FROM user WHERE user_uuid="{id}"').fetchall()
    dbconn.commit()
    dbconn.close()

    downloadPath = []
    for file in query:
        fstatus = file[DBSCHEMA["status"]]
        if(fstatus == "Done"):
            downloadPath.append(os.path.join(PATHBASE, file[DBSCHEMA['processed_path']]))
    
    print(downloadPath)

    zip_file_path = '/tmp/downloaded_files.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in downloadPath:
            zipf.write(file_path, os.path.basename(file_path))

    return send_file(zip_file_path, as_attachment=True)



if __name__ == '__main__':
    # Set the secret key to some random bytes. Keep this really secret!
    app.secret_key = bytes(os.getenv('SECRET_KEY'), 'utf-8')
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)