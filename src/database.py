import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# Database Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(
    exist_ok=True
)

DATABASE_PATH = DATA_DIR / "wastewise.db"


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# Initialize Database
# ============================================================

def initialize_database():
    """
    Create the predictions table if it does not exist.

    Also upgrades older databases by adding the new
    feedback-related columns.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Create table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommendation TEXT,
            feedback TEXT,
            actual_category TEXT,
            feedback_timestamp TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Check existing columns
    # --------------------------------------------------------

    cursor.execute(
        """
        PRAGMA table_info(predictions)
        """
    )

    existing_columns = {
        row["name"]
        for row in cursor.fetchall()
    }

    # --------------------------------------------------------
    # Upgrade old database
    # --------------------------------------------------------

    if "feedback" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN feedback TEXT
            """
        )

    if "actual_category" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN actual_category TEXT
            """
        )

    if "feedback_timestamp" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN feedback_timestamp TEXT
            """
        )

    connection.commit()

    connection.close()


# ============================================================
# Save Prediction
# ============================================================

def save_prediction(
    image_name,
    category,
    confidence,
    recommendation,
):
    """
    Save a new AI prediction into the database.

    Returns:
        int: ID of the newly created prediction.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp,
            image_name,
            category,
            confidence,
            recommendation
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            image_name,
            category,
            float(confidence),
            recommendation,
        ),
    )

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


# ============================================================
# Save User Feedback
# ============================================================

def save_feedback(
    prediction_id,
    feedback,
    actual_category,
):
    """
    Save user feedback for an existing prediction.

    feedback should be:
        "correct"
        or
        "incorrect"

    actual_category:
        Correct category selected by the user.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE predictions
        SET
            feedback = ?,
            actual_category = ?,
            feedback_timestamp = ?
        WHERE id = ?
        """,
        (
            feedback,
            actual_category,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            prediction_id,
        ),
    )

    connection.commit()

    connection.close()


# ============================================================
# Get Prediction History
# ============================================================

def get_predictions(limit=100):
    """
    Return recent predictions.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp,
            image_name,
            category,
            confidence,
            recommendation,
            feedback,
            actual_category,
            feedback_timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# Get Total Predictions
# ============================================================

def get_total_predictions():
    """
    Return total number of predictions.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM predictions
        """
    )

    result = cursor.fetchone()

    connection.close()

    return result["total"]


# ============================================================
# Get Average Confidence
# ============================================================

def get_average_confidence():
    """
    Return average AI confidence.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT AVG(confidence) AS average
        FROM predictions
        """
    )

    result = cursor.fetchone()

    connection.close()

    average = result["average"]

    if average is None:
        return 0.0

    return float(average)


# ============================================================
# Get Category Distribution
# ============================================================

def get_category_distribution():
    """
    Return prediction count grouped by category.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            category,
            COUNT(*) AS count
        FROM predictions
        GROUP BY category
        ORDER BY count DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return {
        row["category"]: row["count"]
        for row in rows
    }


# ============================================================
# Get Feedback Statistics
# ============================================================

def get_feedback_statistics():
    """
    Return feedback statistics.

    Returns:
        total_feedback
        correct
        incorrect
        accuracy
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_feedback,

            SUM(
                CASE
                    WHEN feedback = 'correct'
                    THEN 1
                    ELSE 0
                END
            ) AS correct_predictions,

            SUM(
                CASE
                    WHEN feedback = 'incorrect'
                    THEN 1
                    ELSE 0
                END
            ) AS incorrect_predictions

        FROM predictions

        WHERE feedback IS NOT NULL
        """
    )

    result = cursor.fetchone()

    connection.close()

    total = result["total_feedback"] or 0

    correct = (
        result["correct_predictions"]
        or 0
    )

    incorrect = (
        result["incorrect_predictions"]
        or 0
    )

    if total > 0:
        accuracy = (
            correct / total
        ) * 100
    else:
        accuracy = 0.0

    return {
        "total_feedback": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
    }


# ============================================================
# Get Incorrect Predictions
# ============================================================

def get_incorrect_predictions(limit=100):
    """
    Return predictions marked incorrect by users.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp,
            image_name,
            category,
            confidence,
            actual_category,
            feedback_timestamp
        FROM predictions
        WHERE feedback = 'incorrect'
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# Clear Predictions
# ============================================================

def clear_predictions():
    """
    Delete all stored predictions.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM predictions
        """
    )

    connection.commit()

    connection.close()