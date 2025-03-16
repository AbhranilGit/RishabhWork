CREATE TABLE LANGUAGE (
    language_code VARCHAR(2) PRIMARY KEY,
    language_name VARCHAR(50) NOT NULL
);

-- Insert supported languages
INSERT INTO LANGUAGE (language_code, language_name) VALUES
('de', 'German'),
('en', 'English'),
('es', 'Spanish'),
('fr', 'French'),
('it', 'Italian'),
('nl', 'Dutch');

-- Create USERS table
CREATE TABLE USERS (
    email VARCHAR(255) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    password VARCHAR(255) NOT NULL,
    org_id INTEGER NOT NULL,
    role_id INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create CAMPAIGN table
CREATE TABLE CAMPAIGN (
    campaign_id SERIAL PRIMARY KEY,
    campaign_name VARCHAR(255) NOT NULL,
    created_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    last_modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create EMAIL table
CREATE TABLE EMAIL (
    common_id UUID DEFAULT gen_random_uuid(),
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    from_user VARCHAR(255) NOT NULL,
    campaign_id INTEGER REFERENCES CAMPAIGN(campaign_id),
    header_image VARCHAR(255),
    email_body TEXT NOT NULL,
    preview VARCHAR(255),
    language_code VARCHAR(2) REFERENCES LANGUAGE(language_code),
    created_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    last_modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create TAG table
CREATE TABLE TAG (
    tag_id SERIAL PRIMARY KEY,
    tag_name VARCHAR(100) NOT NULL,
    created_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(255) NOT NULL REFERENCES USERS(email),
    last_modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create EMAIL_TAG junction table
CREATE TABLE EMAIL_TAG (
    email_id VARCHAR(255) REFERENCES EMAIL(id),
    tag_id INTEGER REFERENCES TAG(tag_id),
    PRIMARY KEY (email_id, tag_id)
);

-- Function to update last modified date
CREATE OR REPLACE FUNCTION update_last_modified_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.last_modified_date = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for EMAIL table to update last_modified_date
CREATE TRIGGER update_email_last_modified
BEFORE UPDATE ON EMAIL
FOR EACH ROW
EXECUTE FUNCTION update_last_modified_column();

-- Trigger for TAG table to update last_modified_date
CREATE TRIGGER update_tag_last_modified
BEFORE UPDATE ON TAG
FOR EACH ROW
EXECUTE FUNCTION update_last_modified_column();

-- Function to ensure email id format
CREATE OR REPLACE FUNCTION check_email_id_format()
RETURNS TRIGGER AS $$
BEGIN
   IF NEW.id <> NEW.common_id || NEW.language_code THEN
       RAISE EXCEPTION 'Email id must be in the format common_id || language_code';
   END IF;
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to ensure email id format before insert or update
CREATE TRIGGER ensure_email_id_format
BEFORE INSERT OR UPDATE ON EMAIL
FOR EACH ROW
EXECUTE FUNCTION check_email_id_format();

-- Function to update last_modified_by
CREATE OR REPLACE FUNCTION update_last_modified_by()
RETURNS TRIGGER AS $$
BEGIN
   NEW.last_modified_by = current_user;
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for EMAIL table to update last_modified_by
CREATE TRIGGER update_email_last_modified_by
BEFORE UPDATE ON EMAIL
FOR EACH ROW
EXECUTE FUNCTION update_last_modified_by();

-- Trigger for TAG table to update last_modified_by
CREATE TRIGGER update_tag_last_modified_by
BEFORE UPDATE ON TAG
FOR EACH ROW
EXECUTE FUNCTION update_last_modified_by();

-- Function to set created_by and creation_time
CREATE OR REPLACE FUNCTION set_creation_info()
RETURNS TRIGGER AS $$
BEGIN
   NEW.created_by = current_user;
   NEW.creation_time = CURRENT_TIMESTAMP;
   NEW.last_modified_by = current_user;
   NEW.last_modified_date = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for EMAIL table to set creation info
CREATE TRIGGER set_email_creation_info
BEFORE INSERT ON EMAIL
FOR EACH ROW
EXECUTE FUNCTION set_creation_info();

-- Trigger for TAG table to set creation info
CREATE TRIGGER set_tag_creation_info
BEFORE INSERT ON TAG
FOR EACH ROW
EXECUTE FUNCTION set_creation_info();