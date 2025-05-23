-- To execute this SQL code, follow these steps:

-- 1. Open terminal/command prompt and connect to PostgreSQL:
psql -U postgres

-- 2. If you need to create a new database (if it doesn't exist):
CREATE DATABASE milkquality;

-- 3. Connect to the database:
\c milkquality

-- 4. Execute the table creation statements:
CREATE TABLE IF NOT EXISTS milk_sample (
    sample_id SERIAL PRIMARY KEY,
    sample_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    milk_quality VARCHAR(64),
    action_needed VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS ph_sensor (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    ph_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ec_sensor (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    ec_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS protein_sensor (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    protein_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turbidity_sensor (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    turbidity_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scc_sensor (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    scc_value FLOAT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES milk_sample(sample_id) ON DELETE CASCADE
);

-- 5. Verify the database and tables were created:
\l    -- Lists all databases
\dt   -- Lists all tables in current database

-- Alternative execution methods:
-- Method 1: Execute from file
-- psql -U postgres -f milkquality.sql

-- Method 2: Execute individual commands
-- psql -U postgres -c "CREATE DATABASE milkquality;"
-- psql -U postgres -d milkquality -c "CREATE TABLE IF NOT EXISTS milk_sample..."
