import basicSsl from "@vitejs/plugin-basic-ssl";
import dts from "vite-plugin-dts";

export default {
  plugins: [basicSsl(), dts()],
  build: {
    lib: {
      entry: "src/index.ts",
      name: "strahl",
    },
  },
};
