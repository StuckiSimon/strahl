import { RefObject } from "react";
import useEvent from "./useEvent";

export type TypewriterState = {
  currentText: string;
  currentTextIndex: number;
};

const FULL_SHOW_PERIOD_MS = 1400;

function useTypewriter(writeTarget: RefObject<HTMLElement>) {
  return useEvent((suggestions: string[], typewriterState: TypewriterState) => {
    let timeout: NodeJS.Timeout;
    let state: "write" | "delete" | "shift" = "write";

    let textIndex = 0;
    const animationLoop = () => {
      if (state === "write") {
        textIndex = textIndex + 1;
      } else {
        textIndex = textIndex - 1;
      }
      const textToDisplay = typewriterState.currentText.substring(0, textIndex);
      if (writeTarget.current) {
        if (textToDisplay.length === 0) {
          writeTarget.current.innerHTML = "&nbsp;";
        } else {
          writeTarget.current.textContent = textToDisplay;
        }
      }
      if (
        state === "write" &&
        textIndex === typewriterState?.currentText.length
      ) {
        state = "shift";
      } else if (state === "delete" && textIndex === 0) {
        state = "write";
        const nextIndex =
          (typewriterState.currentTextIndex + 1) % suggestions.length;
        typewriterState.currentText = suggestions[nextIndex];
        typewriterState.currentTextIndex = nextIndex;
      }

      if (state === "write") {
        // Randomize the speed of typing, based on eyeballing
        timeout = setTimeout(animationLoop, 100 + Math.random() * 80);
      } else if (state === "delete") {
        // Randomize the speed of deleting, based on eyeballing
        timeout = setTimeout(animationLoop, 40 + Math.random() * 20);
      } else if (state === "shift") {
        state = "delete";
        timeout = setTimeout(animationLoop, FULL_SHOW_PERIOD_MS);
      }
    };
    animationLoop();
    return () => {
      clearTimeout(timeout);
    };
  });
}

export default useTypewriter;
