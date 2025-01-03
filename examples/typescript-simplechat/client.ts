import * as readline from "readline";

const model = "codellama";
type Message = {
  role: "assistant" | "user" | "system";
  content: string;
}
const messages: Message[] = [{
  role: "system",
  content: "You are a helpful AI agent."
}]

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

async function chat(messages: Message[]): Promise<Message> {
  const body = {
    model: model,
    messages: messages
  }

  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    body: JSON.stringify(body)
  })

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error("Failed to read response body")
  }
  let content = ""
  let reading = true;
  while (reading) {
    const { done, value } = await reader.read()
    if (done) {
      reading = false;
      break;
    }
    
    try {
      const rawjson = new TextDecoder().decode(value);
      const lines = rawjson.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        if (!line) continue;
        const json = JSON.parse(line);
        if (json.done === false) {
          process.stdout.write(json.message.content);
          content += json.message.content;
        }
      }
    } catch (error) {
      console.error('Error parsing JSON:', error);
      continue;
    }
  }
  
  return { role: "assistant", content: content };
}

async function askQuestion(): Promise<void> {
  return new Promise<void>((resolve) => {
    rl.question("\n\nAsk a question: (press enter alone to quit)\n\n", async (user_input) => {
      if (user_input.trim() === "") {
        rl.close();
        console.log("Thankyou. Goodbye.\n")
        console.log("=======\nHere is the message history that was used in this conversation.\n=======\n")
        messages.forEach(message => {
          console.log(message)
        })
        resolve();
      } else {
        console.log();
        messages.push({ role: "user", content: user_input });
        messages.push(await chat(messages));
        await askQuestion(); // Ask the next question
      }
    });
  });
}

async function main() {
  await askQuestion();

}

main();
