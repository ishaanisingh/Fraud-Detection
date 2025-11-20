import pandas as pd
import mysql.connector
import sys
import time
import re
import numpy as np # Import numpy for NaN handling

# --- CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = 'Ishani151219'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'delta_app'
TABLE_NAME = 'app_transaction'

CSV_FILE_PATH = 'C:/Users/ishis/Downloads/dataset.csv'
CHUNK_SIZE = 5000 

# --- SCRIPT LOGIC ---
try:
    # 1. Connect to MySQL
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        autocommit=False,
        connection_timeout=600
    )
    cursor = connection.cursor()
    print("✅ Successfully connected to the MySQL database.")

    # 2. Read CSV file
    print(f"📂 Reading data from '{CSV_FILE_PATH}'...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"✅ CSV data loaded successfully with {len(df)} rows.")

    # 3. Clean and standardize column names
    def clean_column(col):
        col = col.strip()
        col = col.lower()
        col = re.sub(r'[^0-9a-zA-Z_]+', '_', col) 
        return col.strip('_')

    df.columns = [clean_column(c) for c in df.columns]

    # ### --- NEW FIX 1: HANDLE DATES AND TIMES --- ###
    # Convert date/time columns from strings to the correct format for MySQL
    print("🔄 Converting date and time columns...")
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce', dayfirst=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    # 'time' is just a string, which should be fine if your DB model expects TimeField
    # If 'time' conversion fails, you may need:
    # df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

    # ### --- NEW FIX 2: HANDLE MISSING VALUES (NaN -> None) --- ###
    # Replace all pandas 'NaN' (Not a Number) with 'None' (which becomes SQL NULL)
    # NaT (Not a Time) from failed date conversions will also become None
    print("🔄 Converting NaN/NaT to None for SQL NULL...")
    df = df.where(pd.notnull(df), None)

    # 4. Prepare safe insert query (backtick every column)
    
    # ### --- THIS IS YOUR FIX: Add backticks (`) --- ###
    columns = ', '.join([f"`{col}`" for col in df.columns]) 
    
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_query = f"INSERT IGNORE INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"

    print("\n🧩 Prepared INSERT query successfully.")
    print(f"Columns: {', '.join(df.columns)}")

    # 5. Insert data in chunks
    total_rows = len(df)
    print(f"\n🚀 Starting insertion of {total_rows} rows in chunks of {CHUNK_SIZE}...")

    for start in range(0, total_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_rows)
        chunk = df.iloc[start:end]
        
        # Convert chunk to a list of tuples
        data = [tuple(row) for row in chunk.itertuples(index=False, name=None)]

        try:
            cursor.executemany(insert_query, data)
            connection.commit()
            print(f"✅ Inserted rows {start + 1} to {end} successfully.")
        except mysql.connector.Error as batch_err:
            connection.rollback()
            print(f"❌ Error inserting rows {start + 1}-{end}: {batch_err}")
            # Optional: print the first row of data that failed to help debug
            # print(f"Failing data sample: {data[0]}") 
            time.sleep(2)

    print(f"\n🎉 Success! All {total_rows} rows inserted into '{TABLE_NAME}'.")

except FileNotFoundError:
    print(f"❌ Error: The file '{CSV_FILE_PATH}' was not found.")
    sys.exit(1)

except mysql.connector.Error as db_err:
    print(f"❌ Database error: {db_err}")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("🔒 MySQL connection closed.")