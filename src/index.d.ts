export interface VoiceOptions {
    speakerId?: number;
    sentenceSilence?: number;
    expressiveness?: number;
}
export declare function setVoice(modelPath: string, configPath?: string): Promise<void>;
export declare function streamTextToAudio(text: string, options?: VoiceOptions): AsyncGenerator<Buffer>;
declare const _default: {
    setVoice: typeof setVoice;
    streamTextToAudio: typeof streamTextToAudio;
};
export default _default;
//# sourceMappingURL=index.d.ts.map