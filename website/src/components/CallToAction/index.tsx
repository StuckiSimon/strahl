import Heading from "@theme/Heading";
import styles from "./styles.module.css";

export default function CallToAction(): JSX.Element {
  return (
    <div className={styles.root}>
      <Heading as="h2">Get Started</Heading>
      <div className={styles.window}>
        <div className={styles.header}>
          <span />
          <span />
          <span />
        </div>

        <code className={styles.code}>
          <span className={styles.indicator}>$</span> npm install{" "}
          <a href="https://www.npmjs.com/package/strahl">strahl</a>
        </code>
      </div>
    </div>
  );
}
