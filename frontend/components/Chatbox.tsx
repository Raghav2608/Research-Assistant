import { ChangeEvent, useState } from "react";

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
    </div>
  );
}
