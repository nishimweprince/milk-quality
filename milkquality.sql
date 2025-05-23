CREATE DATABASE IF NOT EXISTS milkqualitydb;
USE milkqualitydb;
CREATE TABLE IF NOT EXISTS milk_sample (
    sample_id INT AUTO_INCREMENT PRIMARY KEY,
    sample_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ph_sensor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id INT,
    ph_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS ec_sensor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id INT,
    ec_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS protein_sensor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id INT,
    protein_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS turbidity_sensor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id INT,
    turbidity_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS scc_sensor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id INT,
    scc_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);
show databases;
USE milkqualitydb;
SHOW TABLES;

