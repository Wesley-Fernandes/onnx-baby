// onnx-baby/src/index.ts
import * as ort from 'onnxruntime-node';
import { execSync } from 'node:child_process';
import * as fs from 'node:fs';

const BOS = '^';
const EOS = '$';
const PAD = '_';

interface VoiceConfig {
  audio: { sample_rate: number };
  espeak: { voice: string };
  inference: { noise_scale?: number; length_scale?: number; noise_w?: number };
  num_speakers: number;
  phoneme_id_map: Record<string, number[]>;
  [key: string]: any;
}

let voiceModel: ort.InferenceSession | null = null;
let voiceConfig: VoiceConfig | null = null;

export interface VoiceOptions {
  speakerId?: number;
  sentenceSilence?: number;
  expressiveness?: number;
}

export async function setVoice(modelPath: string, configPath?: string): Promise<void> {
  if (voiceModel) return;
  configPath ??= `${modelPath}.json`;
  voiceConfig = JSON.parse(fs.readFileSync(configPath, 'utf8')) as VoiceConfig;
  voiceModel = await ort.InferenceSession.create(modelPath);
}

export async function* streamTextToAudio(
  text: string,
  options: VoiceOptions = {}
): AsyncGenerator<Buffer> {
  if (!voiceModel || !voiceConfig) throw new Error('Chame setVoice() primeiro');

  const { speakerId = 0, sentenceSilence = 0.32, expressiveness = 0.17 } = options;
  const sentences = text.match(/[^.!?;:]+[.!?;:]?/g) || [text];

  for (let sentence of sentences) {
    sentence = sentence.trim();
    if (!sentence) continue;

    const noiseScale = (voiceConfig.inference.noise_scale ?? 0.667) * (1 + Math.random() * expressiveness * 2 - expressiveness);
    const lengthScale = (voiceConfig.inference.length_scale ?? 1.0) * (0.90 + Math.random() * 0.20);
    const noiseWScale = voiceConfig.inference.noise_w ?? 0.8;

    const floatAudio = await generateSentence(sentence, speakerId, noiseScale, lengthScale, noiseWScale);
    yield floatToInt16Buffer(floatAudio);

    if (sentenceSilence > 0) {
      const silenceSamples = Math.floor(voiceConfig.audio.sample_rate * sentenceSilence);
      yield Buffer.alloc(silenceSamples * 2, 0);
    }
  }
}

async function generateSentence(
  sentence: string,
  speakerId: number,
  noiseScale: number,
  lengthScale: number,
  noiseWScale: number
): Promise<Float32Array> {
  const phonemes = textToPhonemes(sentence);
  const phonemeIds = phonemesToIds(voiceConfig!.phoneme_id_map, phonemes);

  const feeds: Record<string, ort.Tensor> = {
    input: new ort.Tensor('int64', BigInt64Array.from(phonemeIds.map(BigInt)), [1, phonemeIds.length]),
    input_lengths: new ort.Tensor('int64', BigInt64Array.from([BigInt(phonemeIds.length)]), [1]),
    scales: new ort.Tensor('float32', new Float32Array([noiseScale, lengthScale, noiseWScale]), [3]),
  };
  if (voiceConfig!.num_speakers > 1) {
    feeds.sid = new ort.Tensor('int64', BigInt64Array.from([BigInt(speakerId)]));
  }

  const result = await voiceModel!.run(feeds);
  return result.output!.data as Float32Array;
}

function textToPhonemes(text: string): string[][] {
  const voice = voiceConfig!.espeak.voice;
  const output = execSync(`espeak-ng -v ${voice} --ipa=2 -q "${text.replace(/"/g, '\\"')}"`, { encoding: 'utf8' }).trim();
  return [Array.from(output.replace(/\s+/g, '').normalize('NFD'))];
}

function phonemesToIds(idMap: Record<string, number[]>, phonemesList: string[][]): number[] {
  const ids: number[] = [];
  for (const phonemes of phonemesList) {
    ids.push(idMap[BOS]![0], idMap[PAD]![0]);
    for (const p of phonemes) if (idMap[p]) ids.push(idMap[p]![0], idMap[PAD]![0]);
    ids.push(idMap[EOS]![0]);
  }
  return ids;
}

function floatToInt16Buffer(floatArray: Float32Array): Buffer {
  const int16 = new Int16Array(floatArray.length);
  for (let i = 0; i < floatArray.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(floatArray[i] * 32767)));
  }
  return Buffer.from(int16.buffer);
}

export default { setVoice, streamTextToAudio };