import os
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProductMatch:
    """Data class to store product match information"""
    product_id: str
    confidence: float
    product_type: str
    attributes: Dict
    similarity_score: float


class ProductSeeker:
    """
    Main class for processing images and finding matching products
    """

    def __init__(self, db_connection=None, image_storage_path: str = "./image_storage"):
        """
        Initialize ProductSeeker with necessary configurations

        Args:
            db_connection: Database connection object (optional)
            image_storage_path: Path to store/access reference images
        """
        self.db_connection = db_connection
        self.image_storage_path = Path(image_storage_path)
        self.image_storage_path.mkdir(exist_ok=True)

        # Initialize support components
        self.supported_product_types = {
            'book': ['title', 'author', 'isbn'],
            'electronics': ['brand', 'model', 'specs'],
            'toy': ['name', 'age_range', 'category']
        }

    def process_image_path(self, image_path: str) -> List[ProductMatch]:
        """
        Process an image path and return potential product matches

        Args:
            image_path: Path to the image file

        Returns:
            List of ProductMatch objects
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)

            # Extract text from image
            extracted_text = self._extract_text_from_image(image)

            # Detect product type
            product_type = self._detect_product_type(image, extracted_text)

            # Extract relevant attributes
            attributes = self._extract_attributes(image, extracted_text, product_type)

            # Search for matching products
            matches = self._find_matching_products(product_type, attributes, image)

            return matches

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for analysis"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            # Basic preprocessing
            image = cv2.resize(image, (800, 800))  # Standardize size
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert to PIL Image for tesseract
            pil_image = Image.fromarray(image)

            # Extract text
            extracted_text = pytesseract.image_to_string(pil_image)

            return extracted_text.strip()

        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ""

    def _detect_product_type(self, image: np.ndarray, text: str) -> str:
        """Detect product type from image and text"""
        # Simplified implementation - could be enhanced with ML model
        keywords = {
            'book': ['book', 'isbn', 'author', 'edition'],
            'electronics': ['device', 'electronic', 'gadget', 'battery'],
            'toy': ['toy', 'game', 'play']
        }

        text = text.lower()
        for product_type, type_keywords in keywords.items():
            if any(keyword in text for keyword in type_keywords):
                return product_type

        return 'unknown'

    def _extract_attributes(self, image: np.ndarray, text: str, product_type: str) -> Dict:
        """Extract relevant attributes based on product type"""
        attributes = {}

        if product_type in self.supported_product_types:
            # Extract attributes based on product type
            expected_attributes = self.supported_product_types[product_type]

            # Simple attribute extraction (could be enhanced)
            for attribute in expected_attributes:
                # Extract attribute value using specific logic per attribute
                value = self._extract_specific_attribute(text, attribute)
                if value:
                    attributes[attribute] = value

        return attributes

    def _extract_specific_attribute(self, text: str, attribute: str) -> Optional[str]:
        """Extract specific attribute value from text"""
        # Simplified implementation - could be enhanced with NLP
        text_lines = text.split('\n')
        for line in text_lines:
            if attribute.lower() in line.lower():
                return line.split(':')[-1].strip()
        return None

    def _find_matching_products(self, product_type: str, attributes: Dict,
                                image: np.ndarray) -> List[ProductMatch]:
        """Find matching products based on extracted information"""
        matches = []

        # Simulate database search (replace with actual database query)
        mock_products = self._mock_database_search(product_type, attributes)

        for product in mock_products:
            similarity_score = self._calculate_similarity(product, attributes)

            match = ProductMatch(
                product_id=product['id'],
                confidence=similarity_score,
                product_type=product_type,
                attributes=product,
                similarity_score=similarity_score
            )
            matches.append(match)

        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)

        return matches[:5]  # Return top 5 matches

    def _mock_database_search(self, product_type: str, attributes: Dict) -> List[Dict]:
        """Mock database search - replace with actual database implementation"""
        # Example mock data
        return [
            {'id': '1', 'title': 'Sample Product 1', 'brand': 'Brand A'},
            {'id': '2', 'title': 'Sample Product 2', 'brand': 'Brand B'}
        ]

    def _calculate_similarity(self, product: Dict, attributes: Dict) -> float:
        """Calculate similarity score between product and attributes"""
        # Simplified similarity calculation
        matching_attrs = sum(1 for k, v in attributes.items()
                             if k in product and product[k].lower() == v.lower())
        return matching_attrs / max(len(attributes), 1)


# Example usage and testing
def test_product_seeker():
    """Basic test function"""
    try:
        seeker = ProductSeeker()

        # Test with sample image
        sample_image_path = "sample_product.jpg"

        # Create dummy test image if needed
        if not os.path.exists(sample_image_path):
            img = np.zeros((800, 800, 3), dtype=np.uint8)
            cv2.imwrite(sample_image_path, img)

        results = seeker.process_image_path(sample_image_path)

        print("Test Results:")
        for result in results:
            print(f"Product ID: {result.product_id}")
            print(f"Confidence: {result.confidence}")
            print(f"Type: {result.product_type}")
            print("---")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_product_seeker()
