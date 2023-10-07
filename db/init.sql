-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS translations_db;

-- Use the database
USE translations_db;

-- French Sentences table
CREATE TABLE IF NOT EXISTS translations_db.french_sentences (
    id INT PRIMARY KEY,
    french_sentence TEXT
);

-- English Sentences table
CREATE TABLE IF NOT EXISTS translations_db.english_sentences (
    id INT PRIMARY KEY,
    english_sentence TEXT,
    french_sentence_id INT
);

-- Noun Replacements table
-- CREATE TABLE IF NOT EXISTS translations_db.noun_replacements (
--     id INT PRIMARY KEY,
--     english_noun VARCHAR(255),
--     french_noun VARCHAR(255)
-- );

CREATE TABLE IF NOT EXISTS translations_db.noun_replacements (
  id int NOT NULL,
  english_noun varchar(255),
  french_noun varchar(255),
  PRIMARY KEY (id),
  UNIQUE KEY(english_noun, french_noun)
)
