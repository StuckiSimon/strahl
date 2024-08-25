import React from "react";
import { BindingParams, Pane } from "tweakpane";
import {
  Color,
  convertHexToRGB,
  OpenPBRMaterial,
  PathTracerOptions,
} from "strahl";
import styles from "./styles.module.css";
import clsx from "clsx";
import usePathTracer from "@site/src/hooks/usePathTracer";

type PartialOpenPBRMaterialConfiguration = Partial<
  Record<keyof OpenPBRMaterial, unknown>
>;

type Props = {
  propertiesForConfiguration?: (keyof OpenPBRMaterial)[];
  defaultMaterialProperties?: PartialOpenPBRMaterialConfiguration;
  optionOverrides?: Omit<Partial<PathTracerOptions>, "enableDenoise">;
  focusOptions?: (keyof typeof rendererOptionConfiguration)[];
};

function convertNormalizedToHex(c: number) {
  var hex = Math.round(c * 255).toString(16);
  return hex.padStart(2, "0");
}

function convertRGBToHex(c: Color) {
  return (
    "#" +
    convertNormalizedToHex(c[0]) +
    convertNormalizedToHex(c[1]) +
    convertNormalizedToHex(c[2])
  );
}

const defaultColor = "#f20089";

const materialOptionConfiguration: Partial<
  Record<
    keyof OpenPBRMaterial,
    {
      configKey: string;
      value: unknown;
      convertToPaneValue?: (value: unknown) => unknown;
      convertToMaterialValue?: (value: unknown) => unknown;
      bindingParams?: BindingParams;
    }
  >
> = {
  oBaseColor: {
    configKey: "baseColor",
    value: convertHexToRGB(defaultColor),
    convertToPaneValue: (value) =>
      convertRGBToHex(value as ReturnType<typeof convertHexToRGB>),
    convertToMaterialValue: (value) => convertHexToRGB(value as string),
  },
  oBaseWeight: {
    configKey: "baseWeight",
    value: 1.0,
    bindingParams: { min: 0, max: 1 },
  },
  oBaseDiffuseRoughness: {
    configKey: "baseDiffuseRoughness",
    value: 0.0,
    bindingParams: { min: 0, max: 1 },
  },
  oSpecularWeight: {
    configKey: "specularWeight",
    value: 0.0,
    bindingParams: { min: 0, max: 1 },
  },
  oSpecularColor: {
    configKey: "specularColor",
    value: [1.0, 1.0, 1.0],
    convertToPaneValue: (value) =>
      convertRGBToHex(value as ReturnType<typeof convertHexToRGB>),
    convertToMaterialValue: (value) => convertHexToRGB(value as string),
  },
  oSpecularRoughness: {
    configKey: "specularRoughness",
    value: 0.3,
    bindingParams: { min: 0, max: 1 },
  },
  oBaseMetalness: {
    configKey: "baseMetalness",
    value: 0.0,
    bindingParams: { min: 0, max: 1 },
  },
  oEmissionLuminance: {
    configKey: "emissionLuminance",
    value: 0.0,
    bindingParams: { min: 0, max: 1000 },
  },
  oEmissionColor: {
    configKey: "emissionColor",
    value: [1.0, 1.0, 1.0],
    convertToPaneValue: (value) =>
      convertRGBToHex(value as ReturnType<typeof convertHexToRGB>),
    convertToMaterialValue: (value) => convertHexToRGB(value as string),
  },
};

const rendererOptionConfiguration = {
  targetSamples: {
    bindingParams: {
      min: 0,
      max: 10_000,
      step: 1,
    },
  },
  maxRayDepth: {
    bindingParams: {
      min: 1,
      max: 10,
      step: 1,
    },
  },
  size: {
    bindingParams: {
      min: 64,
      max: 2048,
      step: 64,
    },
  },
  denoiseThreshold: {
    bindingParams: {
      min: 0.0,
      max: 10.0,
      step: 0.01,
    },
  },
  denoiseSigma: {
    bindingParams: {
      min: 0.0,
      max: 10.0,
      step: 0.01,
    },
  },
  denoiseKSigma: {
    bindingParams: {
      min: 0.0,
      max: 10.0,
      step: 0.01,
    },
  },
  clearColor: {
    bindingParams: {
      format: "rgb",
    },
  },
  skyPower: {
    bindingParams: {
      min: 0.0,
      max: 10.0,
      step: 0.1,
    },
  },
  skyColor: {
    bindingParams: {
      format: "rgb",
    },
  },
  sunPower: {
    bindingParams: {
      min: 0.0,
      max: 6.0,
      step: 0.1,
    },
  },
  sunAngularSize: {
    bindingParams: {
      min: 0.0,
      max: 180.0,
      step: 0.01,
    },
  },
  sunColor: {
    bindingParams: {
      format: "rgb",
    },
  },
  sunLatitude: {
    bindingParams: {
      min: -90,
      max: 90,
      step: 1,
    },
  },
  sunLongitude: {
    bindingParams: {
      min: 0,
      max: 360,
      step: 1,
    },
  },
} satisfies Partial<
  Record<
    string,
    {
      bindingParams?: BindingParams;
    }
  >
>;

export default function TracerExperiment({
  propertiesForConfiguration = [],
  defaultMaterialProperties,
  optionOverrides,
  focusOptions,
}: Props): JSX.Element {
  const defaultMaterial = {
    ...Object.fromEntries(
      Object.entries(materialOptionConfiguration).map(([key, { value }]) => [
        key,
        value,
      ]),
    ),
    ...defaultMaterialProperties,
  };
  const buildMaterial = (overrides: PartialOpenPBRMaterialConfiguration) => {
    const material = new OpenPBRMaterial();
    const allConfiguredMaterialKeys = Object.entries(
      materialOptionConfiguration,
    ).map(([key]) => key);
    for (const key of allConfiguredMaterialKeys) {
      material[key] = overrides[key] ?? defaultMaterial[key];
    }
    return material;
  };

  const [materialMap, setMaterialMap] = React.useState({
    floor: (() => {
      let m = new OpenPBRMaterial();
      m.oSpecularWeight = 0.0;
      m.oBaseColor = [0.4, 0.4, 0.4];
      return m;
    })(),
    sphere: buildMaterial({}),
  });

  const defaultOptions = {
    targetSamples: 250,
    size: 512,
    clearColor: convertHexToRGB("#1B1B1D"),
    enableDenoise: {
      type: "gaussian",
      threshold: 4.0,
      kSigma: 1.0,
      sigma: 0.07,
    },
    viewProjectionConfiguration: {
      matrixWorldContent: [
        -0.45178184301411944, 4.163336342344336e-17, 0.8921284472108064, 0,
        0.18290622579667423, 0.9787573022265018, 0.09262535237781978, 0,
        -0.8731772322315672, 0.20502229961225985, -0.44218477786341664, 0,
        -3.67881274400709, 0.6362064645963488, -1.879628578827991, 1,
      ],
      fov: 38.6701655,
      cameraTargetDistance: 4,
      aspect: 1,
    },
    maxRayDepth: 3,
    environmentLightConfiguration: {
      sky: {
        power: 0.5,
        color: [0.8, 0.8, 1.0],
      },
      sun: {
        power: 1.0,
        angularSize: 35,
        latitude: 40,
        longitude: 160,
        color: [1.0, 1.0, 0.9],
      },
    },
    ...optionOverrides,
  } satisfies PathTracerOptions;

  const [options, setOptions] =
    React.useState<Parameters<typeof usePathTracer>[2]>(defaultOptions);

  const paneContainerId = React.useId();
  const paneRef = React.useRef<Pane | null>(null);
  React.useEffect(() => {
    let materialParams = Object.fromEntries(
      Object.entries(materialOptionConfiguration).map(
        ([key, { convertToPaneValue, configKey }]) => [
          configKey,
          convertToPaneValue
            ? convertToPaneValue(defaultMaterial[key])
            : defaultMaterial[key],
        ],
      ),
    );
    const PARAMS = {
      ...materialParams,
      ...defaultOptions,
      ...{
        clearColor:
          typeof defaultOptions.clearColor === "boolean"
            ? defaultOptions.clearColor
            : convertRGBToHex(defaultOptions.clearColor),
        denoiseKSigma: defaultOptions.enableDenoise.kSigma,
        denoiseSigma: defaultOptions.enableDenoise.sigma,
        denoiseThreshold: defaultOptions.enableDenoise.threshold,
        skyPower: defaultOptions.environmentLightConfiguration.sky.power,
        skyColor: convertRGBToHex(
          defaultOptions.environmentLightConfiguration.sky.color,
        ),
        sunPower: defaultOptions.environmentLightConfiguration.sun.power,
        sunAngularSize:
          defaultOptions.environmentLightConfiguration.sun.angularSize,
        sunColor: convertRGBToHex(
          defaultOptions.environmentLightConfiguration.sun.color,
        ),
        sunLatitude: defaultOptions.environmentLightConfiguration.sun.latitude,
        sunLongitude:
          defaultOptions.environmentLightConfiguration.sun.longitude,
      },
    };

    const pane = new Pane({
      container: document.getElementById(`pane-${paneContainerId}`),
    });

    for (const property of propertiesForConfiguration) {
      pane.addBinding(
        PARAMS,
        materialOptionConfiguration[property].configKey,
        materialOptionConfiguration[property].bindingParams,
      );
    }
    const focusConfigurableOptions = focusOptions ?? [];
    const putawayConfigurableOptions = Object.keys(
      rendererOptionConfiguration,
    ).filter((key) => !focusConfigurableOptions.includes(key));

    console.log(
      putawayConfigurableOptions,
      Object.keys(focusConfigurableOptions),
    );

    for (const key of focusConfigurableOptions) {
      pane.addBinding(
        PARAMS,
        key,
        rendererOptionConfiguration[key].bindingParams,
      );
    }

    const rendererSettings = pane.addFolder({
      title: "Renderer",
    });

    for (const key of putawayConfigurableOptions) {
      rendererSettings.addBinding(
        PARAMS,
        key,
        rendererOptionConfiguration[key].bindingParams,
      );
    }

    rendererSettings.expanded = false;

    paneRef.current = pane;

    function convertPaneToOpenPBRMaterial(): PartialOpenPBRMaterialConfiguration {
      return Object.fromEntries(
        Object.entries(materialOptionConfiguration).map(
          ([key, { convertToMaterialValue, configKey }]) => [
            key,
            convertToMaterialValue
              ? convertToMaterialValue(PARAMS[configKey])
              : PARAMS[configKey],
          ],
        ),
      );
    }

    pane.on("change", (ev) => {
      setMaterialMap((prev) => {
        const newMaterialMap = { ...prev };

        newMaterialMap.sphere = buildMaterial(convertPaneToOpenPBRMaterial());
        return newMaterialMap;
      });

      setOptions((current) => {
        return {
          ...current,
          targetSamples: PARAMS.targetSamples,
          maxRayDepth: PARAMS.maxRayDepth,
          size: PARAMS.size,
          clearColor:
            typeof PARAMS.clearColor === "boolean"
              ? PARAMS.clearColor
              : convertHexToRGB(PARAMS.clearColor),
          enableDenoise: {
            type: "gaussian",
            threshold: PARAMS.denoiseThreshold,
            kSigma: PARAMS.denoiseKSigma,
            sigma: PARAMS.denoiseSigma,
          },
          environmentLightConfiguration: {
            sky: {
              power: PARAMS.skyPower,
              color: convertHexToRGB(PARAMS.skyColor),
            },
            sun: {
              power: PARAMS.sunPower,
              angularSize: PARAMS.sunAngularSize,
              latitude: PARAMS.sunLatitude,
              longitude: PARAMS.sunLongitude,
              color: convertHexToRGB(PARAMS.sunColor),
            },
          },
        };
      });
    });

    return () => {
      pane.dispose();
    };
  }, []);

  const canvas = usePathTracer(
    "https://stuckisimon.github.io/strahl-sample-models/sphere-plane/demo1.glb",
    materialMap,
    options,
  );

  return (
    <div className={styles.container}>
      <div className="container">
        <div className="row">
          <div id={`pane-${paneContainerId}`} />
          <div className={clsx("col", styles.wrapper)}>{canvas}</div>
        </div>
      </div>
    </div>
  );
}
