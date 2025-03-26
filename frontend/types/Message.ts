import Paper from "./Paper";

export enum Sender {
  User = "user",
  Bot = "bot",
}

export default interface Message {
  message: string;
  sender: Sender;
  papers?: Paper[];
}
