"use client"

import { ChangeEvent, useState } from "react";
import SendButton from "./SendButton";

export default function Chatbox() {
  const [chatInput, setChatInput] = useState<string>("");

  return (
    <div>
      <input
        type="text"
        value={chatInput}
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          setChatInput(e.target.value);
        }}
      />
      <SendButton chatInput={chatInput} setChatInput={setChatInput}/>
    </div>
  );
}
