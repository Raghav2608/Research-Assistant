import Chatbox from "@/components/Chatbox";
import Chatlog from "@/components/Chatlog"
export default function Home() {
  return (
    <div className="flex flex-col-reverse items-center w-full">
      <Chatbox />
      <Chatlog /> 
    </div>
  );
}
