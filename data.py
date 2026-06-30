import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "service_center_data.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_service_center_data.csv")


def create_directories() -> None:
    """Create required project directories."""
    folders = ["data", "models", "logs", "chatbot", "utils", "pages", "notebooks"]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def generate_random_date(start_year: int = 2023, end_year: int = 2026) -> datetime:
    """Generate a random date between two years."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 6, 30)

    total_days = (end_date - start_date).days
    random_days = random.randint(0, total_days)

    return start_date + timedelta(days=random_days)


def get_complaint_templates() -> dict:
    """Complaint templates based on complaint type."""
    return {
        "Battery Issue": [
            "Battery drains very fast",
            "Device shuts down at 20 percent battery",
            "Battery is not charging properly",
            "Battery backup is very poor",
            "Phone heats while charging"
        ],
        "Display Issue": [
            "Screen is flickering",
            "Display has black lines",
            "Touch screen is not working",
            "Screen brightness is very low",
            "Cracked screen replacement needed"
        ],
        "Software Issue": [
            "Device is running very slow",
            "Apps are crashing frequently",
            "Operating system update failed",
            "Device stuck on logo screen",
            "Software reset required"
        ],
        "Charging Issue": [
            "Device is not charging",
            "Charging port is loose",
            "Fast charging not working",
            "Charger disconnects frequently",
            "Charging takes too much time"
        ],
        "Camera Issue": [
            "Camera is blurry",
            "Front camera not opening",
            "Camera app crashes",
            "Back camera not focusing",
            "Camera shows black screen"
        ],
        "Speaker Issue": [
            "Speaker sound is very low",
            "No sound from speaker",
            "Speaker produces crackling noise",
            "Microphone is not working",
            "Audio output is distorted"
        ],
        "Network Issue": [
            "WiFi is not connecting",
            "Mobile network signal is weak",
            "Bluetooth not pairing",
            "Internet disconnects frequently",
            "SIM not detected"
        ],
        "Motherboard Issue": [
            "Device is completely dead",
            "Motherboard needs replacement",
            "Device restarts automatically",
            "Power issue in motherboard",
            "System does not turn on"
        ]
    }


def generate_dataset(rows: int = 2000) -> pd.DataFrame:
    """Generate realistic service center dataset."""

    first_names = [
        "Ranish", "Arun", "Priya", "Kumar", "Divya", "Rahul", "Sneha",
        "Vijay", "Karthik", "Meena", "Anjali", "David", "John", "Alex"
    ]

    last_names = [
        "A", "Kumar", "Raj", "Ravi", "Devi", "Prakash", "S", "M", "Nair",
        "Sharma", "Das", "Bose"
    ]

    products = {
        "Laptop": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
        "Mobile": ["Samsung", "Apple", "OnePlus", "Vivo", "Oppo"],
        "TV": ["Samsung", "LG", "Sony", "Mi"],
        "Refrigerator": ["LG", "Samsung", "Whirlpool", "Godrej"],
        "Washing Machine": ["LG", "Samsung", "Whirlpool", "Bosch"],
        "Air Conditioner": ["LG", "Samsung", "Voltas", "Blue Star"]
    }

    technicians = [
        "Rahul", "Priya", "Arun", "David", "Alex", "John", "Meena", "Karthik"
    ]

    parts_map = {
        "Battery Issue": ["Battery", "Charging IC", "No Part"],
        "Display Issue": ["Display Panel", "Touch Screen", "Screen Glass"],
        "Software Issue": ["No Part", "OS Reinstall", "Software Reset"],
        "Charging Issue": ["Charging Port", "USB Board", "Adapter"],
        "Camera Issue": ["Camera Module", "Camera Lens", "No Part"],
        "Speaker Issue": ["Speaker", "Microphone", "Audio IC"],
        "Network Issue": ["Network IC", "WiFi Module", "SIM Slot"],
        "Motherboard Issue": ["Motherboard", "Power IC", "Processor Board"]
    }

    priority_map = {
        "Battery Issue": ["Medium", "High"],
        "Display Issue": ["Medium", "High"],
        "Software Issue": ["Low", "Medium"],
        "Charging Issue": ["Medium", "High"],
        "Camera Issue": ["Low", "Medium"],
        "Speaker Issue": ["Low", "Medium"],
        "Network Issue": ["Medium", "High"],
        "Motherboard Issue": ["High"]
    }

    complaint_templates = get_complaint_templates()
    complaint_types = list(complaint_templates.keys())

    data = []

    for i in range(1, rows + 1):
        customer_id = f"CUST{i:05d}"
        customer_name = f"{random.choice(first_names)} {random.choice(last_names)}"

        product_category = random.choice(list(products.keys()))
        brand = random.choice(products[product_category])

        purchase_date = generate_random_date(2023, 2025)
        service_request_date = purchase_date + timedelta(days=random.randint(10, 900))

        warranty_status = (
            "Under Warranty"
            if (service_request_date - purchase_date).days <= 365
            else "Out of Warranty"
        )

        complaint_type = random.choice(complaint_types)
        complaint = random.choice(complaint_templates[complaint_type])
        priority = random.choice(priority_map[complaint_type])
        technician = random.choice(technicians)
        parts_replaced = random.choice(parts_map[complaint_type])

        base_repair_time = {
            "Low": random.randint(1, 2),
            "Medium": random.randint(2, 4),
            "High": random.randint(4, 7)
        }

        repair_time = base_repair_time[priority]

        if complaint_type == "Motherboard Issue":
            repair_time += random.randint(2, 4)

        if parts_replaced == "No Part":
            repair_cost = random.randint(0, 500)
        elif warranty_status == "Under Warranty":
            repair_cost = random.randint(0, 1500)
        else:
            repair_cost = random.randint(800, 12000)

        status = random.choice(["Open", "In Progress", "Closed"])

        if status == "Closed":
            customer_rating = random.randint(3, 5)
        else:
            customer_rating = random.randint(1, 4)

        if customer_rating >= 4:
            sentiment = "Positive"
            feedback = random.choice([
                "Good service and quick repair",
                "Technician explained the issue clearly",
                "Satisfied with the service",
                "Repair completed on time"
            ])
        elif customer_rating == 3:
            sentiment = "Neutral"
            feedback = random.choice([
                "Service was okay",
                "Repair took average time",
                "Issue solved but delay happened",
                "Normal service experience"
            ])
        else:
            sentiment = "Negative"
            feedback = random.choice([
                "Repair took too much time",
                "Not satisfied with service",
                "Problem happened again",
                "Poor communication from service center"
            ])

        data.append({
            "Customer_ID": customer_id,
            "Customer_Name": customer_name,
            "Product": product_category,
            "Product_Category": product_category,
            "Brand": brand,
            "Purchase_Date": purchase_date.date(),
            "Service_Request_Date": service_request_date.date(),
            "Warranty_Status": warranty_status,
            "Complaint": complaint,
            "Complaint_Type": complaint_type,
            "Priority": priority,
            "Technician": technician,
            "Repair_Time": repair_time,
            "Parts_Replaced": parts_replaced,
            "Repair_Cost": repair_cost,
            "Status": status,
            "Customer_Rating": customer_rating,
            "Feedback": feedback,
            "Sentiment": sentiment
        })

    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame) -> None:
    """Save raw and clean datasets."""
    df.to_csv(RAW_DATA_PATH, index=False)

    clean_df = df.drop_duplicates()
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print("Dataset generated successfully!")
    print(f"Raw dataset saved at: {RAW_DATA_PATH}")
    print(f"Clean dataset saved at: {CLEAN_DATA_PATH}")
    print(f"Shape: {clean_df.shape}")
    print("\nFirst 5 rows:")
    print(clean_df.head())


def main() -> None:
    """Main function."""
    create_directories()
    df = generate_dataset(rows=2000)
    save_dataset(df)


if __name__ == "__main__":
    main()