import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./money.css";
import App from "./App";

createRoot(document.getElementById("money")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
