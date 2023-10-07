import logging
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from src.textSummarizer.logging import logger
import mysql.connector


class DatabaseUtility:
    def __init__(self, config_path):
        try:
            with open(config_path) as yaml_file:
                content = yaml.safe_load(yaml_file)
                self.config = content
            self.connection = None
        except BoxValueError:
            raise ValueError("yaml file is empty")
        except Exception as e:
            raise e
        
    def connect(self):
        """
        Connect to the MySQL database.
        """
        try:
            if not self.connection:
                self.connection = mysql.connector.connect(
                    host=self.config['database']['host'],
                    user=self.config['database']['user'],
                    password=self.config['database']['password'],
                    database=self.config['database']['name']
                )
                logger.info("Connected to the database.")
                return self.connection
        except mysql.connector.Error as err:
            logger.error(f"Error connecting to the database: {err}")
            return Exception(err)

    def execute_query(self, query, params=None):
        """
        Execute a SQL query.
        :param query: SQL query.
        :param params: Query parameters.
        :return: Query Result.
        """
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except mysql.connector.Error as err:
            logger.error(f"Error executing query: {err}")
            return Exception(err)

    def execute_many(self, query, data=None):
        """
        Execute a SQL query.
        :param query: SQL query.
        :param data: List of tuples containing data to be inserted.
        :return: Query Result.
        """
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor()
            cursor.executemany(query, data)  # Using executemany for multiple insertions
            self.connection.commit()  # Commit the changes to the database
            cursor.close()
        except mysql.connector.Error as err:
            logging.error(f"Error executing query: {err}")
            raise Exception(err)


    def close_connection(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")
