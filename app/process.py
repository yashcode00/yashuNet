from multiprocessing import Pool,cpu_count
from multiprocessing.process import current_process
import sys, traceback, os, datetime
import sqlite3
from dotenv import load_dotenv
import time

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

load_dotenv()

def process(*file):
    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()

    time.sleep(2)    
    ppath = os.path.join(__file__, f'converted/{file[1]}')

    dbcurs.execute(f"""UPDATE user SET processed_path="{ppath}" WHERE file_uuid="{file[0]}"; """)
    dbcurs.execute(f"""UPDATE user SET status="Done" WHERE file_uuid="{file[0]}"; """)

    dbconn.commit()
    dbconn.close()

def main():

    dbconn = sqlite3.connect(os.getenv('DATABASE_PATH'))
    dbcurs = dbconn.cursor()

    query = dbcurs.execute(f'SELECT * FROM user WHERE status="Pending" ORDER BY created_at').fetchall()
    # print(query)
    dispatcher = []
    for file in query:
        dispatcher.append( [file[DBSCHEMA['file_uuid']], file[DBSCHEMA['name']],
            file[DBSCHEMA['path']]] )

    dbconn.commit()
    dbconn.close()

    p = Pool(processes = 8) #max(len(data),cpu_count())
    result_mult = p.starmap(process, dispatcher)
    p.close()
    p.join()
