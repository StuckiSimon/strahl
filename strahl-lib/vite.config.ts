import basicSsl from "@vitejs/plugin-basic-ssl";

export default {
  plugins: [basicSsl()],
  build: {
    lib: {
      entry: "src/main.ts",
      name: "strahl",
    },
  },
};
