from flask import Flask, jsonify
import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

app = Flask(__name__)

def get_snowflake_connection():
    """Establishes a Snowflake connection."""
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    )

@app.route("/data", methods=["GET"])
def get_data():
    """Fetches data from Snowflake and returns JSON."""
    conn = get_snowflake_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM your_table LIMIT 10")
        data = cur.fetchall()
        return jsonify({"data": data})
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    app.run(debug=True)
