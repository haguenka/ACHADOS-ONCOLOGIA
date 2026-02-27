# ACHADOS ONCO - Dashboard para Render

Projeto web do dashboard do app `pdf_tumor_findings_miner.py`, adaptado para Streamlit.

## Arquivos

- `dashboard_onco_render.py`: dashboard web
- `pages/1_Mineracao_Onco.py`: mineracao oncologica (upload PDF -> grava no banco)
- `requirements.txt`: dependencias Python
- `render.yaml`: configuracao pronta para Render (Blueprint)

## Banco de dados

O dashboard le a tabela `patients` do SQLite `tumor_findings_patients.db`.
A pagina de mineracao grava na mesma tabela `patients`.

Configurar caminho pelo ambiente:

- Variavel: `DB_PATH`
- Exemplo: `DB_PATH=/data/tumor_findings_patients.db`

Se `DB_PATH` nao for definido, o app tenta:

1. `./tumor_findings_patients.db`
2. `../tumor_findings_patients.db`

## Deploy no Render

### Opcao A (recomendado)

1. Suba **somente esta pasta** `ACHADOS ONCO` para um repositorio.
2. No Render: `New +` -> `Blueprint`.
3. Selecione o repositorio e aplique.
4. Em `Environment`, configure `DB_PATH` para o caminho real do banco.

### Opcao B (monorepo)

Se este repositorio tiver muitos projetos:

1. Crie um Web Service no Render (nao Blueprint).
2. Defina `Root Directory` como `ACHADOS ONCO`.
3. Build command: `pip install -r requirements.txt`
4. Start command:
   `streamlit run dashboard_onco_render.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
5. Configure `DB_PATH`.

## Rodar local

```bash
cd "ACHADOS ONCO"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard_onco_render.py
```

## Fluxo no Render (mesmo banco)

1. Abra o app publicado.
2. Va para a pagina `Mineracao Onco` (menu lateral do Streamlit).
3. Envie os PDFs e clique em `Minerar e salvar no banco`.
4. Volte para a pagina principal do dashboard para visualizar os dados atualizados.
