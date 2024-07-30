import clsx from "clsx";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Heading from "@theme/Heading";
import { faGithub } from "@fortawesome/free-brands-svg-icons";

import styles from "./index.module.css";
import Button from "../components/Button";
import useTypewriter, { TypewriterState } from "../hooks/useTypewriter";
import React from "react";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  const typeWriterSuggestions = [
    "fast",
    "efficient",
    "powerful",
    "beautiful",
    "simple",
    "open-source",
    "understandable",
    "accessible",
  ];

  const typewriterRef = React.useRef<HTMLHeadingElement>(null);
  const typewriterState = React.useRef<TypewriterState | null>({
    currentText: typeWriterSuggestions[0],
    currentTextIndex: 0,
  });
  const startTypewriter = useTypewriter(typewriterRef);

  React.useEffect(() => {
    return startTypewriter(typeWriterSuggestions, typewriterState.current);
  }, []);

  return (
    <header className={clsx("hero", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          Path Tracing <span className={styles.heroMark}>made</span>
        </Heading>
        <div className={styles.typewriterContainer}>
          <p
            className={clsx("hero__subtitle", styles.heroTypewriter)}
            ref={typewriterRef}
          ></p>
          <div className={styles.cursorContainer}>
            <div className={styles.cursor}></div>
          </div>
        </div>
        <p className={clsx("hero__subtitle", styles.heroText)}>
          strahl is an open-source path tracing library built using WebGPU.
        </p>
        <div className={styles.buttons}>
          <Button to="/blog">Get Started</Button>
          <Button
            to="https://github.com/StuckiSimon/strahl"
            icon={faGithub}
            variant="secondary"
          >
            Github
          </Button>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="strahl"
      description="strahl – path tracing library using WebGPU and OpenPBR"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
