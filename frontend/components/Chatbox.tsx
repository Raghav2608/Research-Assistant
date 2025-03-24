"use client";

import { ChangeEvent, useState, KeyboardEvent, Key } from "react";
import SendButton from "./SendButton";
import Message, { Sender } from "@/types/Message";
import ModeButton from "./ModeButton";

export interface ChatboxProps {
  addMessage: (msg: Message) => void;
}

export default function Chatbox({ addMessage }: ChatboxProps) {
  const [chatInput, setChatInput] = useState<string>("");
  const [mode, setMode] = useState<string>("fast");
  const queryurl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/query`;

  async function send(): Promise<void> {
    // Check if the input is empty
    if (chatInput == "") return;

    const user_query = chatInput;
    setChatInput("");

    addMessage({ message: user_query, sender: Sender.User });

    // Get a response from the API
    try {
      const res = await fetch(queryurl, {
        method: "POST",
        credentials: "include", // Ensure cookies are sent/received
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_query, mode }),
      });

      if (res.ok) {
        const data = await res.json();
        addMessage({
          message: data.answer,
          sender: Sender.Bot,
          papers: data.papers,
        });
      } else {
        addMessage({
          message: "Sorry! I am unable to respond to this query",
          sender: Sender.Bot,
        });
      }
    } catch (error) {
      console.error("Error processing the request", error);
    }
  }

  function handleEnter(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="sticky bottom-10 pb-2 bg-light w-full flex flex-row justify-center items-center gap-0">
      <ModeButton mode={mode} setMode={setMode} />
      <input
        placeholder="Type your query here..."
        type="text"
        value={chatInput}
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          setChatInput(e.target.value);
        }}
        onKeyDown={handleEnter}
        className="w-3/5 ml-5 px-10 py-5 border border-light rounded-full bg-lightest text-white text-3xl placeholder-white placeholder-opacity-50 placeholder- focus:outline-none focus:ring-2 focus:ring-primary"
      />
      <SendButton
        chatInput={chatInput}
        setChatInput={setChatInput}
        send={send}
      />
    </div>
  );
}
