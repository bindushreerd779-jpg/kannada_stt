import React, {useState} from 'react';
import {View, Text, TouchableOpacity, Alert} from 'react-native';
import {
  startRecordingNative,
  stopRecordingNative,
  transcribeNative,
} from './WhisperNative';

export default function App() {
  const [recording, setRecording] = useState(false);
  const [audioPath, setAudioPath] = useState('');
  const [text, setText] = useState('');

  const startRec = async () => {
    try {
      const path = await startRecordingNative();
      console.log('Native recording path:', path);
      setAudioPath(path);
      setRecording(true);
      setText('');
    } catch (e) {
      Alert.alert('Error', String(e?.message || e));
    }
  };

  const stopRec = async () => {
    try {
      const path = await stopRecordingNative();
      console.log('Stopped path:', path);
      setAudioPath(path);
      setRecording(false);
    } catch (e) {
      Alert.alert('Error', String(e?.message || e));
    }
  };

  const transcribe = async () => {
    try {
      if (!audioPath) {
        Alert.alert('Error', 'No recorded WAV found');
        return;
      }

      console.log('Sending file to Whisper:', audioPath);
      const out = await transcribeNative(audioPath);
      console.log('Result:', out);
      setText(out);
    } catch (e) {
      Alert.alert('Error', String(e?.message || e));
    }
  };

  return (
    <View style={{flex: 1, padding: 24, justifyContent: 'center'}}>
      <TouchableOpacity
        onPress={startRec}
        disabled={recording}
        style={{padding: 14, backgroundColor: '#222', marginBottom: 12}}>
        <Text style={{color: 'white'}}>Start Recording</Text>
      </TouchableOpacity>

      <TouchableOpacity
        onPress={stopRec}
        disabled={!recording}
        style={{padding: 14, backgroundColor: '#222', marginBottom: 12}}>
        <Text style={{color: 'white'}}>Stop Recording</Text>
      </TouchableOpacity>

      <TouchableOpacity
        onPress={transcribe}
        style={{padding: 14, backgroundColor: '#222', marginBottom: 12}}>
        <Text style={{color: 'white'}}>Transcribe</Text>
      </TouchableOpacity>

      <Text>Path: {audioPath}</Text>
      <Text>Text: {text}</Text>
    </View>
  );
}