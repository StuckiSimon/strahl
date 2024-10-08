import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Modern Technology",
    description: (
      <>
        Strahl leverages WebGPU, the latest web graphics API built for years to
        come.
      </>
    ),
  },
  {
    title: "Fully Open-Source",
    description: (
      <>
        Strahl is built out in public under MIT license, your feedback is
        welcome.
      </>
    ),
  },
  {
    title: "Using OpenPBR",
    description: (
      <>
        Strahl supports Physically-based Rendering based on the OpenPBR surface
        shading model.
      </>
    ),
  },
];

function Feature({ title, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
