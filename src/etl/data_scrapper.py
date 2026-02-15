import csv
import os
import re
import time
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


class FlipkartScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def start_driver(self):
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--headless=new")  # REMOVE if you want visible browser
        
        return uc.Chrome(options=options, use_subprocess=True)

    def close_popup(self, driver):
        """Close Flipkart login popup if it appears."""
        try:
            btn = driver.find_element(By.CSS_SELECTOR, "button._2KpZ6l._2doB4z")
            btn.click()
            time.sleep(1)
        except:
            pass

    def parse_product_cards(self, driver):
        """Extract product card elements WITHOUT relying on CSS classes."""
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # select <a> tags whose href contains /p/itm (product page)
        links = soup.select("a[href*='/p/itm']")

        product_cards = []

        for link in links:
            parent = link.parent

            # climb up to find the complete product block
            for _ in range(5):
                if parent and parent.get_text(strip=True):
                    product_cards.append(parent)
                parent = parent.parent

        # make unique
        unique_cards = []
        seen = set()

        for card in product_cards:
            txt = card.get_text(strip=True)
            if txt in seen:
                continue
            seen.add(txt)
            unique_cards.append(card)

        return unique_cards

    def extract_product_details(self, card):
        """Extract product fields from the dynamic card using patterns, not classes."""

        text = card.get_text(" ", strip=True)

        # Product title: longest text fragment
        parts = sorted(text.split(" "), key=len, reverse=True)
        title = parts[0] if parts else "N/A"

        # Price: ₹xxxx
        price_match = re.search(r"₹\s?[\d,]+", text)
        price = price_match.group(0) if price_match else "N/A"

        # Rating
        rating_match = re.search(r"\b[0-5]\.?[0-9]?\b", text)
        rating = rating_match.group(0) if rating_match else "N/A"

        # Total Reviews
        review_match = re.search(r"[\d,]+\s+(?:Reviews|Ratings|ratings)", text)
        total_reviews = review_match.group(0).split()[0] if review_match else "N/A"

        # Product ID
        link_tag = card.find("a", href=True)
        href = link_tag["href"]
        product_id_match = re.search(r"itm[0-9A-Za-z]+", href)
        product_id = product_id_match.group(0) if product_id_match else "N/A"

        product_url = "https://www.flipkart.com" + href

        return product_id, title, rating, total_reviews, price, product_url

    def get_reviews(self, url, count=2):
        """Scrape reviews from product page."""
        driver = self.start_driver()
        try:
            driver.get(url)
            time.sleep(3)
            self.close_popup(driver)

            soup = BeautifulSoup(driver.page_source, "html.parser")

            reviews = []
            for block in soup.find_all("div"):
                txt = block.get_text(" ", strip=True)
                if len(txt) > 40:  # review-like
                    reviews.append(txt)
                if len(reviews) >= count:
                    break

            if not reviews:
                return "No reviews found"

            return " || ".join(reviews[:count])

        except:
            return "No reviews found"
        finally:
            driver.quit()

    def scrape_flipkart_products(self, query, max_products=3, review_count=2):
        driver = self.start_driver()
        results = []
        try:
            url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(3)

            self.close_popup(driver)

            cards = self.parse_product_cards(driver)

            if not cards:
                print("⚠ No product cards detected!")
                return []

            for card in cards[:max_products]:
                product_id, title, rating, total_reviews, price, product_url = \
                    self.extract_product_details(card)

                reviews = self.get_reviews(product_url, review_count)

                results.append([product_id, title, rating, total_reviews, price, reviews])

            return results

        except Exception as e:
            print("Error:", e)
            return []
        finally:
            driver.quit()

    def save_to_csv(self, data, filename="product_reviews.csv"):
        # If filename includes folder like data/..., DO NOT prefix with output_dir
        # Resolve output path (absolute or relative)
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # e.g., "data/products.csv"
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # Default output inside data directory
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)

