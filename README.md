# OmniVoice — app e vozes do Canal IA 51+

Esta pasta guarda o app do OmniVoice e as vozes usadas na narração do
Canal IA 51+ / portal ia51mais.com.

## Por que isso existe

Antes, o app e as vozes ficavam no Google Drive. Resultado: toda sessão nova
do Colab parava numa tela pedindo autorização do Drive, e essa tela só passa
com um clique humano — o que impedia a narração de rodar sozinha.

Com os arquivos aqui, o Colab baixa tudo sem pedir autorização nenhuma.

## O que tem aqui

- `omnivoice_app_v14.py` — o app (versão 6.0). É cópia exata do v13 (5.9) com
  uma única diferença: lê as vozes desta pasta em vez do Google Drive.
- `vozes_salvas/` — a voz **Ari** (a oficial do canal) e a **elaNando** (teste).

## Como o Colab usa

O notebook "OmniVoice - Gerador de Voz (ABRIR ESTE)" baixa este repositório
para `/content/omnivoice_dados` e roda o `omnivoice_app_v14.py` de lá.

## Onde as coisas são salvas agora

- Narrações: `/content/narracoes`, dentro do Colab. **Somem quando o Colab
  desliga** — baixe antes pelo botão de download do app.
- Vozes novas que você clonar: também somem ao desligar. Para guardar uma voz
  nova de vez, ela precisa ser enviada para este repositório.

## Backup

O app antigo (v11, v12, v13) continua no Google Drive, intocado.
Backup completo em `F:\backupgeral\backup_omnivoice\`.
