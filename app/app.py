from flask import Flask, flash, jsonify, request,send_file, redirect, url_for, redirect, render_template, send_from_directory
from dotenv import load_dotenv
import os, sys
import subprocess
import datetime
import threading
import logging
import uuid
import time
# from werkzeug.utils import secure_filename
import json
from flask_cors import CORS
import sqlite3

logger = logging.getLogger('webserver')
logging.basicConfig(level=logging.INFO,
    format='%(name)-10s %(levelname)-8s [%(asctime)s] %(message)s',
)

load_dotenv()
PATHBASE = os.path.abspath(os.path.dirname(__file__))
logger.info(f"Base Path : {PATHBASE}")

# if 'uploads' not in os.listdir():
#     print("Creating Upload directory.....")
#     os.makedirs('uploads')
# if 'converted' not in os.listdir():
#     print("Creating COnverted directory.....")
#     os.makedirs('converted')

app = Flask(__name__)
CORS(app)
# sets max payload limit 
app.config['MAX_CONTENT_LENGTH'] = 200 * 1000 * 1000

# setup and start the flask app
app = Flask(__name__)
app.app_context().push()


# try :
#     dbconn = sqlite3.connect('instance/database.db')
#     dbcurs = dbconn.cursor()
#     dbcurs.execute("SELECT * FROM user")
#     dbconn.close()
# except sqlite3.OperationalError :
#     logger.warning("DB : Creating new database")
#     open('instance/database.db','w').close()
#     dbconn = sqlite3.connect('instance/database.db')
#     dbcurs = dbconn.cursor()
#     dbcurs.execute("""
# CREATE TABLE user (
#         user_uuid VARCHAR(36) NOT NULL, 
#         file_uuid VARCHAR(36) NOT NULL, 
#         name VARCHAR(100) NOT NULL, 
#         "desiredExtension" VARCHAR(10) NOT NULL, 
#         "originalExtension" VARCHAR(10) NOT NULL, 
#         path VARCHAR(100) NOT NULL, 
#         created_at DATETIME DEFAULT (CURRENT_TIMESTAMP), 
#         status VARCHAR(100) DEFAULT 'Pending' NOT NULL, 
#         converted_file_path VARCHAR(100) DEFAULT 'NaN' NOT NULL, 
#         PRIMARY KEY (user_uuid, file_uuid)
# );
# """)
#     dbconn.commit()
#     dbconn.close()

# DBSCHEMA = {
#     "user_uuid" : 0,
#     "file_uuid" : 1,
#     "name" : 2,
#     "desiredExtension" : 3,
#     "originalExtension" : 4,
#     "path" : 5,
#     "created_at" : 6,
#     "status" : 7,
#     "converted_file_path" : 8,
# }


# ***************************************
# BEGIN server route definitions
# ***************************************

@app.route('/')
def landing_page():
    """Home page. User can upload files from here"""
    return render_template('landing.html')
    

# @app.route('/upload', methods=['POST','GET'])
# def upload_page():
#     dbconn = sqlite3.connect('instance/database.db')
#     dbcurs = dbconn.cursor()
#     if request.method == 'POST':
#         files = {}
#         if len(request.files) == 0:
#             flash('No file part')
#             return render_template('landing.html')

#         for f in request.files.keys():
#             # extract name of file
#             filename = secure_filename(request.files[f].filename)
#             uuid_here = f.split("_")[1]
#             temp = {}
#             # getting src_type, target and name
#             for k in request.form.keys():
#                 if k.split("_")[1] == uuid_here:
#                     temp[k.split("_")[0]] = request.form[k]
#             temp['uuid']=uuid_here
#             files[request.files[f]]=temp

#         # unique user id
#         user_uuid=str(uuid.uuid1())
#         # files_descp = []
#         print(files.keys())
#         for f in files.keys():
#             # extract name of file
#             filename = files[f]['name']
#             # target extension
#             originalExtension = files[f]['srctype']
#             # print(originalExtension)
#             desiredExtension = files[f]['target']
#             # new name
#             filename = os.path.splitext(filename)[0] + "_" + str(files[f]['uuid']) + "." + originalExtension.lower()
#             # saving files locally
#             path =  os.path.join("uploads",filename)
#             f.save(path)  

#             id = str(files[f]['uuid'])  # file id
#             dbcurs.execute("""
#             INSERT INTO user(user_uuid,file_uuid,name,
#             "desiredExtension","originalExtension",path) VALUES (?,?,?,?,?,?) 
#             """, (user_uuid, id, filename, desiredExtension, originalExtension, path))
#             dbconn.commit()
#             # logger.info(f"Created file {id}")

#         dbconn.commit()
#         dbconn.close()

#         t = threading.Thread(target=spawn,)
#         t.start()
#         # os.system('python3 convert.py')

#         return redirect(url_for('.display_page',user_uuid=user_uuid)) 
#         # user is directed to /display and using AJAX, converted files are displayed
#     else:
#         """Home page. User can upload files from here"""
#         dbconn.commit()
#         dbconn.close()
#         return render_template('landing.html')

#     # print(PATHBASE)

# @app.route('/display')
# def display_page():
#     """Display page to download files"""
#     user_uuid = request.args['user_uuid']
#     print(user_uuid)
#     return render_template('display.html', user_uuid=user_uuid)


# @app.route('/status/<id>')
# def status_check(id):
#     """Return JSON with info about whether the uploaded file has been parsed successfully."""
#     # query = User.query.filter(User.user_uuid == id).all()
#     dbconn = sqlite3.connect('instance/database.db')
#     dbcurs = dbconn.cursor()
#     query = dbcurs.execute(f'SELECT * FROM user WHERE user_uuid="{id}"').fetchall()
#     response = []
#     dbconn.commit()
#     dbconn.close()

#     for file in query:
#         fstatus = file[DBSCHEMA["status"]]
#         if fstatus == 'Done':
#             message = 'File parsed successfully.'
#         elif fstatus == 'Pending':
#             message = 'File parsing pending.'
#         else:
#             message = 'File parsing failed.'

#         response.append({
#             'user_id': file[DBSCHEMA["user_uuid"]],
#             'file_id': file[DBSCHEMA["file_uuid"]],
#             'status': fstatus,
#             'name': file[DBSCHEMA["name"]],
#             'message': message,
#             'converted_path': file[DBSCHEMA["converted_file_path"]]
#         })
#     return jsonify(response)


# @app.route('/download/<id>')
# def download_file(id):
#     """Download the converted file."""
#     print("download link")
#     dbconn = sqlite3.connect('instance/database.db')
#     dbcurs = dbconn.cursor()
#     query = dbcurs.execute(f'SELECT * FROM user WHERE file_uuid="{id}"').fetchone()
#     if query:
#         filename = os.path.splitext(query[DBSCHEMA["name"]])[0] + '.' + query[DBSCHEMA["desiredExtension"]].lower()
#         dbconn.commit()
#         dbconn.close()
#         return send_file('converted/'+filename, as_attachment=True)
#     else:
#         dbconn.commit()
#         dbconn.close()
#         return '', 404




if __name__ == '__main__':
    # Set the secret key to some random bytes. Keep this really secret!
    app.secret_key = bytes(os.getenv('SECRET_KEY'), 'utf-8')
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)