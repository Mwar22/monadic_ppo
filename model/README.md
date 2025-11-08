# robô thor pronto para ser utilizado no mujoco
Primeiramente, foi necessário editar o urdf para que os arquivos mesh fossem
escritos relativos ao arquivo urdf. A estrutura para dar certo para o urdf foi:

```
thor_robot/
├── barkour_vb_mjx.xml
├── meshes
│   ├── art1_3d.001.obj
│   ├── ...
│   └── base.001.obj
└── thor_robot.urdf
```

onde as referencias dentro de urdf eram relativas, ex: `meshes/art1_3d.001.obj`

## Teste do urdf
Basta executar a ferramenta de visualização do mujoco, arrastando o arquivo .urdf e vendo se renderiza:
```bash
python -m mujoco.viewer
```
## Conversão para o formato xml do Mujoco (MJCF)
Para isto basta instalar a aplicação `urdf2mjcf`:

```bash
pip install urdf2mjcf
```

Em seguida, basta executar o comando para seu .urdf, no nosso caso:
```bash
urdf2mjcf thor_robot.urdf
```

Obteremos assim, um arquivo .xml equivalente para a utilização no mujoco: `thor_robot.xml`

Para funcionar no MJX, por conta do MJX nao suportar o solver PGS, editamos o  `thor_robot.xml` e removemos a linha 
com a tag <option> comentando ela:
```xml
<!---
  <option iterations="50" timestep="0.001" solver="auto" gravity="0 0 -9.81" />
-->
```
    