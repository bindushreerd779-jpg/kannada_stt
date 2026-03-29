## Kannada Speech-to-Text App

This project implements a Kannada Speech-to-Text system using Whisper models integrated into a React Native mobile application.

## Features
- Kannada speech recognition
- On-device inference
- React Native UI

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   npm install --legacy-peer-deps
   , cd android
   , .\gradlew clean
   
4.  Due to GitHub size limitations, model files are hosted on Google Drive.
    Download model files:
    https://drive.google.com/drive/folders/1_J4adPwJ6s9sjZWP1yy7PGlO3wlv3Vy3?usp=sharing

5. After downloading, place models in:
   android/app/src/main/assets/

6. Run the app:
   npx react-native run-android
