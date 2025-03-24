import { Dispatch, SetStateAction } from "react";
import { Send } from "lucide-react";

export interface SendButtonProps {
  chatInput: string;
  setChatInput: Dispatch<SetStateAction<string>>;
  send: () => void;
}

export default function SendButton({ send }: SendButtonProps) {
  return (
    <button
      onClick={send}
      className="w-20 bg-primary rounded-full m-2 h-3/4 hover:bg-dark hover:text-lightest flex items-center justify-center "
    >
      <Send className="w-5 h-5" />
    </button>
  );
}
