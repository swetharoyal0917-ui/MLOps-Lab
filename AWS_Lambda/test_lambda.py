from lambda_function import lambda_handler

event = {
    "body": """
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
}

print(lambda_handler(event, None))