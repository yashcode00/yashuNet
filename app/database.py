CREATE_TABLE = """
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
"""

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

INSERT_QUERY = """
            INSERT INTO user(user_uuid,file_uuid,name,path,expiry) VALUES (?,?,?,?,?) 
            """

def get_user_query(id):
    return f'SELECT * FROM user WHERE user_uuid="{id}"'