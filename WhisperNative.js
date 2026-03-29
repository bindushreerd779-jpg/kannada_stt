import {NativeModules, PermissionsAndroid, Platform} from 'react-native';

const {WhisperModule} = NativeModules;

async function ensureMicPermission() {
  if (Platform.OS !== 'android') return true;

  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone Permission',
      message: 'App needs microphone access for recording.',
      buttonPositive: 'OK',
    },
  );

  return granted === PermissionsAndroid.RESULTS.GRANTED;
}

export async function startRecordingNative() {
  const ok = await ensureMicPermission();
  if (!ok) {
    throw new Error('Microphone permission denied');
  }
  return await WhisperModule.startRecording();
}

export async function stopRecordingNative() {
  return await WhisperModule.stopRecording();
}

export async function transcribeNative(path) {
  return await WhisperModule.transcribe(path);
}