"use client";

import { ChangeEvent, useState, KeyboardEvent, Key } from "react";
import SendButton from "./SendButton";

export default function Chatbox() {
  const [chatInput, setChatInput] = useState<string>("");

  function send(): void {
    // Check if the input is empty
    if (chatInput == "") return;

    console.log(chatInput);
    setChatInput("");
  }

  function handleEnter(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="w-full flex flex-row justify-center gap-0">
      <input
        placeholder="Type your query here..."
        type="text"
        value={chatInput}
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          setChatInput(e.target.value);
        }}
        onKeyDown={handleEnter}
        className="w-3/5 px-10 py-5 border border-light rounded-full bg-lightest text-white text-3xl placeholder-white placeholder-opacity-50 placeholder- focus:outline-none focus:ring-2 focus:ring-primary"
      />
      <SendButton
        chatInput={chatInput}
        setChatInput={setChatInput}
        send={send}
      />
    </div>
  );
}
