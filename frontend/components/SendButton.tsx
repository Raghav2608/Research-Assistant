import { Dispatch, SetStateAction } from "react";

export interface SendButtonProps {
  chatInput: string;
  setChatInput: Dispatch<SetStateAction<string>>;
}
export default function SendButton({
  chatInput,
  setChatInput,
}: SendButtonProps) {
  function send() {
    console.log(chatInput);
    setChatInput("");
  }

  return <button onClick={send}>Send</button>;
}
