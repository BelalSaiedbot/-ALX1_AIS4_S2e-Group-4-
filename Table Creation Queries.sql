CREATE TABLE Customer (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,  
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL
); 
CREATE TABLE Product (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(255)
);
 CREATE TABLE DateDimension (
    date_id INT PRIMARY KEY,        
    full_date DATE NOT NULL UNIQUE,  
    year INT,                       
    quarter INT,                    
    month INT,                      
    day_of_month INT,               
    day_of_week INT,                
    week_of_year INT
);
CREATE TABLE CustomerOrder (
    order_id INT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    order_date_id INT,  
    arrival_date_id INT,  
    FOREIGN KEY (customer_id) REFERENCES Customer(customer_id),
    FOREIGN KEY (product_id) REFERENCES Product(product_id),
    FOREIGN KEY (order_date_id) REFERENCES DateDimension(date_id),
    FOREIGN KEY (arrival_date_id) REFERENCES DateDimension(date_id)
);
 CREATE TABLE Review (
    review_id INT PRIMARY KEY,
    product_id INT,
    customer_id INT,
    date_id INT,  
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),  
    review_text TEXT,
    FOREIGN KEY (product_id) REFERENCES Product(product_id),
    FOREIGN KEY (customer_id) REFERENCES Customer(customer_id),
    FOREIGN KEY (date_id) REFERENCES DateDimension(date_id),
);
CREATE TABLE ReviewAnalysis (
    analysis_id INT IDENTITY(1,1) PRIMARY KEY,  
    review_id INT,  
    sentiment_category VARCHAR(10) CHECK (sentiment_category IN ('positive', 'neutral', 'negative')),  
    sentiment_score DECIMAL(3, 2),  
    analysis_date DATETIME DEFAULT GETDATE(),  
    FOREIGN KEY (review_id) REFERENCES Review(review_id)
);









