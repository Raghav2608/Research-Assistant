import Chatbox from "@/components/Chatbox";
import Chatlog from "@/components/Chatlog";
import { useState } from "react";
import Message from "@/types/Message";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);

  function addMessage(msg: Message): void {
    setMessages((prevMessages) => [...prevMessages, msg]);
  }
  return (
    <div className="flex flex-col-reverse items-center w-full">
      <Chatbox addMessage={addMessage} />
      <Chatlog messages={messages} />
    </div>
  );
}
