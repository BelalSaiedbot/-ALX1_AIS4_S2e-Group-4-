SELECT 
    r.review_id, 
    p.product_name, 
    r.rating, 
    r.review_text
FROM Review r
JOIN Product p ON r.product_id = p.product_id
WHERE r.review_text IS NOT NULL AND LEN(CAST(r.review_text AS VARCHAR(MAX))) > 0
ORDER BY r.rating DESC;

SELECT AVG(rating) AS average_rating
FROM Review;



SELECT 
    p.product_id, 
    p.product_name,  
    AVG(r.rating) AS average_rating, 
    COUNT(r.review_id) AS review_count
FROM Review r
JOIN Product p ON r.product_id = p.product_id
GROUP BY p.product_id, p.product_name
ORDER BY average_rating DESC;



SELECT TOP 5
    p.product_id, 
    p.product_name, 
    AVG(r.rating) AS average_rating,
	COUNT(r.review_id) AS review_count
FROM Review r
JOIN Product p ON r.product_id = p.product_id
GROUP BY p.product_id, p.product_name
ORDER BY average_rating DESC
;  


SELECT 
    p.category, 
    AVG(r.rating) AS average_rating, 
    COUNT(r.review_id) AS total_reviews
FROM Review r
JOIN Product p ON r.product_id = p.product_id
GROUP BY p.category
HAVING COUNT(r.review_id) > 0
ORDER BY average_rating DESC;



SELECT 
    d.year, 
    d.month, 
    COUNT(r.review_id) AS review_count, 
    AVG(r.rating) AS average_rating
FROM Review r
JOIN DateDimension d ON r.date_id = d.date_id
GROUP BY d.year, d.month
ORDER BY d.year, d.month;


SELECT TOP 10
    c.customer_id, 
    c.customer_name, 
    COUNT(r.review_id) AS total_reviews, 
    AVG(r.rating) AS average_rating
FROM Customer c
JOIN Review r ON c.customer_id = r.customer_id
GROUP BY c.customer_id, c.customer_name
ORDER BY total_reviews DESC;

