import express, { Request, Response } from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import axios from 'axios';

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/api/chat', async (req: Request, res: Response) => {
  const userInput = req.body.message;
  console.log("USER INPUT: ", userInput)
  try {
    const response = await axios.post('http://localhost:8000/generate', {
      input_text: userInput,
    });
    res.json({ response: response.data });
  } catch (error) {
    console.error('Fehler beim Generieren der Antwort:', error);
    res.status(500).json({ error: 'Fehler beim Generieren der Antwort.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server läuft auf http://localhost:${PORT}`);
});
