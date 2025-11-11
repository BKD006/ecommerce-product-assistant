import csv
import time
import re
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


class FlipkartScraper:
    """
    A web scraper for extracting product details and top reviews from Flipkart.

    This scraper uses `undetected_chromedriver` (a stealth version of Selenium Chrome)
    to bypass bot detection mechanisms and scrape structured data such as:
        - Product ID
        - Title
        - Rating
        - Total reviews count
        - Price
        - Top N reviews (text content)

    Features:
        - Handles popups and scrolling automatically
        - Extracts multiple products per query
        - Saves results to a CSV file
        - Robust against missing or invalid fields

    Example:
        >>> scraper = FlipkartScraper()
        >>> data = scraper.scrape_flipkart_products("iphone 15", max_products=3, review_count=2)
        >>> scraper.save_to_csv(data, "data/flipkart_iphone_reviews.csv")
    """

    def __init__(self, output_dir: str = "data"):
        """
        Initialize the scraper with an output directory.

        Args:
            output_dir (str): Directory to store output files. Defaults to "data".
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Method: get_top_reviews
    # -------------------------------------------------------------------------
    def get_top_reviews(self, product_url: str, count: int = 2) -> str:
        """
        Fetch the top reviews for a given Flipkart product page.

        Args:
            product_url (str): URL of the product page on Flipkart.
            count (int): Number of top reviews to extract. Defaults to 2.

        Returns:
            str: Concatenated review texts separated by ' || ', 
                 or "No reviews found" if none are extracted.
        """
        # Setup Chrome options for stealth browsing
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = uc.Chrome(options=options, use_subprocess=True)

        # Validate product URL
        if not product_url.startswith("http"):
            driver.quit()
            return "No reviews found"

        try:
            driver.get(product_url)
            time.sleep(4)

            # Attempt to close popup if present
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
                time.sleep(1)
            except Exception as e:
                print(f"Popup close skipped: {e}")

            # Scroll multiple times to ensure full content loads
            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            # Parse page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Review blocks may vary depending on page layout
            review_blocks = soup.select("div._27M-vq, div.col.EPCmJX, div._6K-7Co")

            reviews = []
            seen = set()

            # Extract unique reviews
            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break

        except Exception as e:
            print(f"Error extracting reviews: {e}")
            reviews = []

        finally:
            driver.quit()

        return " || ".join(reviews) if reviews else "No reviews found"

    # -------------------------------------------------------------------------
    # Method: scrape_flipkart_products
    # -------------------------------------------------------------------------
    def scrape_flipkart_products(self, query: str, max_products: int = 1, review_count: int = 2):
        """
        Scrape Flipkart for products matching a given search query.

        Args:
            query (str): Product search query (e.g., "laptop", "iphone 15").
            max_products (int): Number of products to scrape. Defaults to 1.
            review_count (int): Number of top reviews to fetch per product. Defaults to 2.

        Returns:
            list[list[str]]: A list of product details in the format:
                [product_id, title, rating, total_reviews, price, top_reviews]
        """
        options = uc.ChromeOptions()
        driver = uc.Chrome(options=options, use_subprocess=True)

        # Construct Flipkart search URL
        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(4)

        # Close initial popup if it appears
        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
        except Exception as e:
            print(f"Popup close skipped: {e}")

        time.sleep(2)
        products = []

        # Identify product listing elements
        items = driver.find_elements(By.CSS_SELECTOR, "div[data-id]")[:max_products]

        for item in items:
            try:
                # Extract product details (title, price, rating, etc.)
                title = item.find_element(By.CSS_SELECTOR, "div.KzDlHZ").text.strip()
                price = item.find_element(By.CSS_SELECTOR, "div.Nx9bqj").text.strip()
                rating = item.find_element(By.CSS_SELECTOR, "div.XQDdHH").text.strip()
                reviews_text = item.find_element(By.CSS_SELECTOR, "span.Wphh3N").text.strip()

                # Extract total review count using regex
                match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                total_reviews = match.group(0) if match else "N/A"

                # Extract product URL and product ID
                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                product_link = href if href.startswith("http") else f"https://www.flipkart.com{href}"
                match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                product_id = match[0] if match else "N/A"

            except Exception as e:
                print(f"Error processing product item: {e}")
                continue

            # Fetch top reviews for the product
            top_reviews = (
                self.get_top_reviews(product_link, count=review_count)
                if "flipkart.com" in product_link
                else "Invalid product URL"
            )

            # Store product details
            products.append([product_id, title, rating, total_reviews, price, top_reviews])

        driver.quit()
        return products

    # -------------------------------------------------------------------------
    # Method: save_to_csv
    # -------------------------------------------------------------------------
    def save_to_csv(self, data: list, filename: str = "product_reviews.csv"):
        """
        Save the scraped product data to a CSV file.

        Args:
            data (list): List of product information (each as a list of values).
            filename (str): Output CSV filename. Defaults to "product_reviews.csv".
                            Supports absolute paths or relative subfolders.

        Returns:
            None
        """
        # Resolve output path (absolute or relative)
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # e.g., "data/products.csv"
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # Default output inside data directory
            path = os.path.join(self.output_dir, filename)

        # Write data to CSV file
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)

        print(f"[INFO] Data successfully saved to {path}")
