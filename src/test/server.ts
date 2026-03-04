import express from "express"
import * as WS from "ws"
import onxxbaby from "../index"
import http from "http"
import cors from "cors"

const app = express()
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "PUT", "DELETE"],
  allowedHeaders: ["Content-Type", "Authorization"],
}))
const server = http.createServer(app)
const wss = new WS.WebSocketServer({ server })

// Carrega modelo uma única vez
async function loadModel() {
  await onxxbaby.setVoice("./src/test/models/model.onnx")
  console.log("Modelo carregado na memória.")
}

wss.on("connection", (ws) => {
  console.log("Cliente conectado")

  ws.on("message", async (message) => {
    try {
      const { text, options } = JSON.parse(message.toString())

      if (!text) {
        ws.send(JSON.stringify({ error: "Texto não enviado" }))
        return
      }

      const stream = onxxbaby.streamTextToAudio(text, {
        sentenceSilence: options?.sentenceSilence ?? 0.35,
        expressiveness: options?.expressiveness ?? 0.2
      })

      for await (const chunk of stream) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(chunk) // envia buffer binário
        }
      }

      ws.send(JSON.stringify({ done: true }))

    } catch (err) {
      console.error(err)
      ws.send(JSON.stringify({ error: "Erro ao processar TTS" }))
    }
  })
})

server.listen(3000, async () => {
  await loadModel()
  console.log("Servidor rodando em http://localhost:3000")
})