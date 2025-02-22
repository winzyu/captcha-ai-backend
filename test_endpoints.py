import requests
import json

def test_random_captcha():
    print("\nTesting random CAPTCHA endpoint...")
    response = requests.get('http://localhost:5000/api/captcha/random')
    
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print(f"Image Path: {data['imagePath']}")
        print(f"AI Prediction: {data['prediction']}")
        print(f"Confidence: {data['confidence']:.2%}")
        return data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def test_verify_captcha(image_file, test_text):
    print(f"\nTesting CAPTCHA verification with text: {test_text}")
    response = requests.post(
        'http://localhost:5000/api/captcha/verify',
        json={
            'text': test_text,
            'imageFile': image_file
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print(f"Correct: {data['correct']}")
        print(f"Actual Text: {data['actualText']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test getting a random CAPTCHA
    captcha_data = test_random_captcha()
    
    if captcha_data:
        # Test verification with correct text
        test_verify_captcha(
            captcha_data['imagePath'],
            captcha_data['prediction']  # Using AI prediction as test input
        )
        
        # Test verification with incorrect text
        test_verify_captcha(
            captcha_data['imagePath'],
            "wrong"
        )
