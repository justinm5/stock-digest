"""
Build a synthetic financial headlines dataset for sentiment model training.
Generates 30,599 samples across 3 classes: positive, negative, neutral.
"""
import csv
import random
import os

random.seed(42)

positive_templates = [
    "{company} reports strong {quarter} earnings, beating analyst estimates",
    "{company} raises full-year guidance after record {quarter} revenue",
    "{company} announces {billion} share buyback program",
    "{company} stock surges as {metric} exceeds expectations",
    "{company} secures major {deal} contract with {partner}",
    "{company} declares {dividend} dividend increase",
    "{company} expands into {market} with new product launch",
    "{company} beats revenue forecasts by {percent}",
    "{company} achieves record {metric} in {quarter}",
    "{company} stock climbs on news of {milestone}",
    "{company} announces strategic partnership with {partner}",
    "{company} raises outlook amid strong demand",
    "{company} reports better-than-expected {metric}",
    "{company} shares jump after {event}",
    "{company} to acquire {target} in accretive deal",
]

negative_templates = [
    "{company} misses {quarter} earnings estimates as {metric} falls short",
    "{company} cuts full-year guidance amid weakening demand",
    "{company} announces {percent} workforce reduction",
    "{company} stock plunges following weak {quarter} results",
    "{company} faces regulatory probe over {issue}",
    "{company} delays {product} launch due to supply issues",
    "{company} reports unexpected loss in {quarter}",
    "{company} slashes dividend as cash flow declines",
    "{company} CEO departs amid {scandal}",
    "{company} downgraded by {firm} to {rating}",
    "{company} warns of slowing growth in {market}",
    "{company} shares tumble on {event}",
    "{company} recalls {product} due to safety concerns",
    "{company} loses major {deal} contract",
    "{company} debt rating cut by {firm}",
]

neutral_templates = [
    "{company} schedules {quarter} earnings call for {date}",
    "{company} to hold annual shareholder meeting on {date}",
    "{company} announces leadership transition plan",
    "{company} reports {quarter} results in line with estimates",
    "{company} maintains guidance for fiscal year",
    "{company} appoints new {role} effective {date}",
    "{company} to present at {conference} on {date}",
    "{company} completes previously announced {deal}",
    "{company} files {document} with SEC",
    "{company} updates investor relations website",
    "{company} confirms prior guidance range",
    "{company} announces quarterly dividend",
    "{company} releases sustainability report",
    "{company} opens new office in {location}",
    "{company} renews partnership with {partner}",
]

companies = [
    "Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "NVIDIA", "JPMorgan",
    "Berkshire Hathaway", "Johnson & Johnson", "Visa", "ExxonMobil", "Procter & Gamble",
    "UnitedHealth", "Mastercard", "Home Depot", "Bank of America", "AbbVie", "Pfizer",
    "Coca-Cola", "PepsiCo", "Disney", "Netflix", "Salesforce", "Adobe", "Oracle",
    "Cisco", "Comcast", "Verizon", "AT&T", "Intel", "AMD", "Broadcom", "Qualcomm",
    "Costco", "Walmart", "Target", "Lowe's", "CVS", "McDonald's", "Starbucks", "Nike",
    "Boeing", "Lockheed Martin", "General Electric", "IBM", "Honeywell", "3M", "UPS",
]

metrics = ["revenue", "profit margin", "EPS", "operating income", "cash flow", "gross margin",
           "net income", "EBITDA", "free cash flow", "subscriber growth", "same-store sales",
           "cloud revenue", "advertising revenue", "iPhone revenue", "services revenue"]
quarters = ["Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter", "third quarter", "fourth quarter"]
partners = ["Microsoft", "Amazon", "Google", "Apple", "Tesla", "Samsung", "IBM", "Oracle", "SAP"]
targets = ["Rivian", "Slack", "Square", "Peloton", "Zoom", "Snap", "Twitter", "Dropbox", "Spotify"]
firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Bank of America", "Citigroup", "UBS"]
ratings = ["sell", "underweight", "neutral", "hold"]
products = ["iPhone", "Model 3", "Azure", "Prime", "Pixel", "Oculus", "Surface", "Echo"]
markets = ["China", "Europe", "India", "Southeast Asia", "Latin America", "North America"]
events = ["earnings miss", "product delay", "guidance cut", "CEO departure", "regulatory news"]
scandals = ["accounting issues", "data breach", "lawsuit", "antitrust scrutiny"]
issues = ["privacy practices", "accounting methods", "antitrust concerns", "safety violations"]
conferences = ["CES", "MWC", "Web Summit", "Davos", " investor day"]
roles = ["CFO", "COO", "CTO", "CEO", "Chief Revenue Officer"]
locations = ["Austin", "Singapore", "London", "Dublin", "Toronto", "Bangalore"]
documents = ["10-K", "10-Q", "8-K", "proxy statement", "annual report"]
deals = ["licensing", "distribution", "supply", "cloud", "defense"]


def format_template(template):
    mapping = {
        "{company}": random.choice(companies),
        "{metric}": random.choice(metrics),
        "{quarter}": random.choice(quarters),
        "{partner}": random.choice(partners),
        "{target}": random.choice(targets),
        "{firm}": random.choice(firms),
        "{rating}": random.choice(ratings),
        "{product}": random.choice(products),
        "{market}": random.choice(markets),
        "{event}": random.choice(events),
        "{scandal}": random.choice(scandals),
        "{issue}": random.choice(issues),
        "{conference}": random.choice(conferences),
        "{role}": random.choice(roles),
        "{location}": random.choice(locations),
        "{document}": random.choice(documents),
        "{deal}": random.choice(deals),
        "{billion}": f"${random.randint(1, 50)} billion",
        "{percent}": f"{random.randint(5, 30)}%",
        "{dividend}": f"{random.randint(5, 25)}%",
        "{date}": f"{random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {random.randint(1, 28)}",
        "{milestone}": random.choice(["record deliveries", "billion-user mark", "million-subscriber milestone"]),
    }
    result = template
    for key, val in mapping.items():
        result = result.replace(key, val)
    return result


def generate_headlines(target_total=30_599):
    samples = []
    per_class = target_total // 3
    remainder = target_total - 3 * per_class

    for label, templates in [("positive", positive_templates), ("negative", negative_templates), ("neutral", neutral_templates)]:
        count = per_class + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        generated = set()
        while len(generated) < count:
            template = random.choice(templates)
            headline = format_template(template)
            generated.add(headline)
        for headline in generated:
            samples.append((headline, label))

    random.shuffle(samples)
    return samples


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples = generate_headlines(30_599)
    out_path = os.path.join(root, "data", "financial_headlines.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "sentiment"])
        writer.writerows(samples)
    print(f"Generated {len(samples)} headlines -> {out_path}")


if __name__ == "__main__":
    main()
