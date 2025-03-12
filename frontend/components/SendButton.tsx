import { Dispatch, SetStateAction } from "react";

export interface SendButtonProps {
  chatInput: string;
  setChatInput: Dispatch<SetStateAction<string>>;
  send: () => void;
}
export default function SendButton({
  chatInput,
  setChatInput,
  send,
}: SendButtonProps) {
  return (
    <button onClick={send} className="w-20 bg-primary rounded-full m-2 h-3/4 hover:bg-dark hover:text-lightest">
      <span className="text-l text-lightest">Send</span>
    </button>
  );
}
