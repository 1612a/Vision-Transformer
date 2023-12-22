# Vision-Transformer
•人工智慧期末報告 11122108林哲賢   使用 Vision Transformer 進行物件偵測
##目錄
•準備資料
•實際操作
#準備資料
•準備一個可使用google colab的帳號
#實際操作
•導入和設定
![螢幕擷取畫面 2023-12-22 194239](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/190019ec-09d3-48d0-a06e-a28db333efac)

•將keras進行升級，並進行RESTART SESSION
![螢幕擷取畫面 2023-12-22 194217](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/f2c5c031-f7fb-41e6-a935-202857b038e2)
![螢幕擷取畫面 2023-12-22 194152](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/5915e3be-ba07-4392-874e-be7c44e5182e)

•準備資料集
![螢幕擷取畫面 2023-12-22 194510](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/f880c1c5-804f-4c73-ae74-366c4705f410)
![螢幕擷取畫面 2023-12-22 194534](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/b91a6ade-16c2-4c37-81e0-f4f941ac2b49)
![螢幕擷取畫面 2023-12-22 194604](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/358b8a28-1ad2-48a9-9289-38a3c69f0a71)

•執行多層感知器(MLP)
![螢幕擷取畫面 2023-12-22 194747](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/03439b1a-e071-4419-adb5-5bc29e8e735d)

•實作補丁建立層
![螢幕擷取畫面 2023-12-22 194821](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/dc05d165-21e7-4178-bccd-60e485780597)

•顯示輸入影像的補丁
![螢幕擷取畫面 2023-12-22 194903](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/d33e7f46-c5b3-41b3-8eb0-b60188a921cd)
![螢幕擷取畫面 2023-12-22 194927](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/c71089b9-ddde-4212-bee2-f2be12c5cc3d)

•執行補丁編碼層
![螢幕擷取畫面 2023-12-22 195002](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/c950a097-0a66-483f-be81-0da1e0c84a98)

•建構VIT模型
![螢幕擷取畫面 2023-12-22 195032](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/d16cc17f-c585-45cb-a03f-ac9b02d65d07)
![螢幕擷取畫面 2023-12-22 195053](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/5ff4c7bd-d2d7-4092-b98f-39106d5f3281)

•運行程式
![螢幕擷取畫面 2023-12-22 195153](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/8597a51c-a84a-411d-be0a-58e199830892)
![螢幕擷取畫面 2023-12-22 195231](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/7c131ab7-e82c-4d96-aa8a-714d016338e5)
![螢幕擷取畫面 2023-12-22 195313](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/5e83f8b0-1651-478f-9eb2-ab6c845a385a)

•評估模型
![螢幕擷取畫面 2023-12-22 195403](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/6ecca497-c22b-4286-982a-07f364c75721)
![螢幕擷取畫面 2023-12-22 195428](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/21f01585-6acf-4afc-989f-f9d8e42e0522)
![螢幕擷取畫面 2023-12-22 195448](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/406618ae-071d-4a3c-ab0e-7989b95d27b5)
![螢幕擷取畫面 2023-12-22 195529](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/5f83b9c0-af92-4e60-ba15-e5beb1fef7fa)
![螢幕擷取畫面 2023-12-22 195542](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/c8764b41-481e-4444-b105-a46f15a98077)
![螢幕擷取畫面 2023-12-22 195553](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/11da82d8-95d0-4d54-9543-0f63ad4ae423)
![螢幕擷取畫面 2023-12-22 195600](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/efa6d799-93a6-47bb-8b14-f264390fc26e)
![螢幕擷取畫面 2023-12-22 195608](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/07d187b2-d31c-4a03-939c-5b2ddae326fa)
![螢幕擷取畫面 2023-12-22 195616](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/19b0aaf6-425a-402c-b887-64c75b6ae5d0)
![螢幕擷取畫面 2023-12-22 195624](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/52a35af3-b726-4deb-af25-ffa9916128d8)
![螢幕擷取畫面 2023-12-22 195632](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/a0c20ede-5244-4aa6-a11f-842dc4af7efd)
![螢幕擷取畫面 2023-12-22 195639](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/5f354306-8756-4d18-981f-16fdd934698f)
![螢幕擷取畫面 2023-12-22 195648](https://github.com/RiceXnoodles/Vision-Transformer/assets/148970977/a34c1b15-785c-48f4-ad1d-b362c9632cda)

##參考資料
https://keras.io/examples/vision/object_detection_using_vision_transformer/
